//! Photon-driven NRC: SPPC query lifetimes and hash-encoded online learning.
use super::{GpuWorld, buffer_entry};
use crate::settings::Settings;
use anyhow::{Result, ensure};
use bytemuck::{Pod, Zeroable};
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Parameters {
    config: [u32; 4],
    radii: [f32; 4],
    low: [f32; 4],
    scale: [f32; 4],
}
pub(super) struct PhotonNrc {
    pub tensor: Option<super::tensor::Cache>,
    pub group: wgpu::BindGroup,
    #[cfg(test)]
    pub data: wgpu::Buffer,
    #[cfg(test)]
    query_buffer: wgpu::Buffer,
    #[cfg(test)]
    scalar_probe: wgpu::ComputePipeline,
    uniform: wgpu::Buffer,
    queries: u32,
    hash_parameters: u32,
    initialized: bool,
    pub batches: u32,
    init: wgpu::ComputePipeline,
    trace: [[wgpu::ComputePipeline; 2]; 3],
    resolve: wgpu::ComputePipeline,
    cooperative: bool,
    guides: wgpu::ComputePipeline,
    passes: Vec<(wgpu::ComputePipeline, u32, u32)>,
}
fn source(queries: u32, log2: u32) -> String {
    let mut text = format!(
        "const SP_QUERIES:u32={queries}u;\nconst SP_BLOCKS:u32={}u;\nconst SP_HASH_ENTRIES:u32={}u;\nconst SP_HASH_PARAMETERS:u32={}u;\n{}\n{}\n{}\n{}\n{}\n{}",
        queries / 256,
        1u32 << log2,
        32u32 << log2,
        include_str!("../shaders/trace.wgsl"),
        include_str!("../shaders/adaptive.wgsl"),
        include_str!("../shaders/nrc.wgsl"),
        include_str!("../shaders/photon_nrc.wgsl"),
        include_str!("../shaders/photon_nrc_model.wgsl"),
        include_str!("../shaders/denoise.wgsl")
    );
    if cfg!(test) {
        text.push_str(
            r#"
@compute @workgroup_size(64)
fn sp_scalar_probe(@builtin(global_invocation_id) id:vec3<u32>){
 let slots=select(1u,2u,params.transport.x==2u);if id.x>=params.size.x*params.size.y*slots{return;}
 let s=neural_queries[id.x];var c=vec3<f32>(0.0);
 if s.label.w>0.0&&any(s.label.xyz!=vec3<f32>(0.0)){c=s.label.xyz*sp_predict(s);}
 for(var j=0u;j<3u;j++){sp.gradients[id.x*3u+j]=c[j];}
}
"#,
        );
    }
    text
}
impl PhotonNrc {
    pub fn new(
        device: &wgpu::Device,
        base: &wgpu::BindGroupLayout,
        size: [u32; 2],
        settings: &Settings,
        trace_group_y: u32,
    ) -> Result<Self> {
        let queries = settings
            .photon_nrc_queries
            .clamp(1024, 32768)
            .next_power_of_two();
        let hash_parameters = 32u32 << settings.photon_nrc_hash_log2.clamp(10, 19);
        let bytes = 4_495_936u64
            + u64::from(hash_parameters) * 20
            + u64::from(queries) * 172
            + u64::from(queries / 64)
            + 8192;
        ensure!(
            bytes <= device.limits().max_storage_buffer_binding_size,
            "Photon NRC exceeds device storage limits"
        );
        let data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SPPC queries, hash encoding and neural optimizer"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SPPC parameters"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SPPC inference queries"),
            size: u64::from(size[0]) * u64::from(size[1]) * 128,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let extra = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Photon NRC"),
            entries: &[
                buffer_entry(0, wgpu::BufferBindingType::Storage { read_only: false }),
                buffer_entry(1, wgpu::BufferBindingType::Uniform),
                buffer_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Photon NRC"),
            layout: &extra,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: query_buffer.as_entire_binding(),
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Photon NRC kernels"),
            bind_group_layouts: &[Some(base), Some(&extra)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SPPC and hash-encoded NRC"),
            source: wgpu::ShaderSource::Wgsl(
                super::ray_query::shader(
                    device,
                    source(queries, settings.photon_nrc_hash_log2.clamp(10, 19)),
                )
                .into(),
            ),
        });
        let cooperative = device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
        let matrix = cooperative.then(|| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("SPPC fused Metal inference"),
                source: wgpu::ShaderSource::Wgsl(
                    super::ray_query::shader(
                        device,
                        format!(
                            "enable wgpu_cooperative_matrix;\n{}\n{}",
                            source(queries, settings.photon_nrc_hash_log2.clamp(10, 19)),
                            include_str!("../shaders/photon_nrc_matrix.wgsl")
                        ),
                    )
                    .into(),
                ),
            })
        });
        let pipeline = |entry: &'static str, mode: usize, guides: bool| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: if entry == "sp_resolve_matrix" {
                    matrix.as_ref().unwrap()
                } else {
                    &shader
                },
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("ACTIVE_MODE", mode as f64),
                        ("TRACE_GROUP_Y", trace_group_y as f64),
                        ("PHOTON_NRC_ENABLED", 1.),
                        ("NRC_QUEUED", u32::from(entry == "accumulate") as f64),
                        ("DENOISE_INPUT", u32::from(guides) as f64),
                    ],
                    ..Default::default()
                },
                cache: None,
            })
        };
        let groups = queries.div_ceil(64);
        let passes = [
            ("sp_prepare", groups, 1),
            ("sp_count", groups, 1),
            ("sp_scan_buckets", queries / 256, 1),
            ("sp_scan_blocks", 1, 1),
            ("sp_scatter", groups, 1),
            ("sp_trace_photons", 0, 2),
            ("sp_finalize", groups, 1),
            (
                "sp_environment_update",
                1024u32.min(queries).div_ceil(64),
                1,
            ),
            ("sp_labels", 2, 1),
            ("sp_train", 128, 1),
            ("sp_apply", 1024, hash_parameters.max(8515).div_ceil(65536)),
            ("sp_finish", 1, 1),
        ]
        .into_iter()
        .map(|(entry, x, y)| (pipeline(entry, 0, false), x, y))
        .collect();
        let tensor = super::tensor::Cache::new(
            device,
            base,
            &extra,
            size[0] * size[1],
            64,
            super::ray_query::shader(
                device,
                source(queries, settings.photon_nrc_hash_log2.clamp(10, 19)),
            ),
        );
        Ok(Self {
            tensor,
            group,
            #[cfg(test)]
            data,
            #[cfg(test)]
            query_buffer,
            #[cfg(test)]
            scalar_probe: pipeline("sp_scalar_probe", 0, false),
            uniform,
            queries,
            hash_parameters,
            initialized: false,
            batches: 0,
            init: pipeline("sp_initialize", 0, false),
            trace: std::array::from_fn(|m| {
                std::array::from_fn(|g| pipeline("accumulate", m, g != 0))
            }),
            resolve: pipeline(
                if cooperative {
                    "sp_resolve_matrix"
                } else {
                    "sp_resolve"
                },
                0,
                false,
            ),
            cooperative,
            guides: pipeline("sp_finish_guides", 0, false),
            passes,
        })
    }
    #[cfg(test)]
    pub fn verify_matrix_predictions(&self, r: &super::Renderer) {
        if !self.cooperative {
            return;
        }
        let count = r.width * r.height;
        let stage = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SPPC matrix equivalence"),
            size: u64::from(count) * 76,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = r.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.scalar_probe);
            pass.set_bind_group(0, &r.bind_group, &[]);
            pass.set_bind_group(1, &self.group, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        self.resolve(
            &mut encoder,
            &r.bind_group,
            [r.width, r.height],
            0,
            false,
            false,
        );
        encoder.copy_buffer_to_buffer(&self.data, 16 + 8515 * 16, &stage, 0, u64::from(count) * 12);
        encoder.copy_buffer_to_buffer(
            &self.query_buffer,
            0,
            &stage,
            u64::from(count) * 12,
            u64::from(count) * 64,
        );
        let bytes = r.read_buffer(encoder, &stage).unwrap();
        let mut checked = 0;
        for i in 0..count as usize {
            for c in 0..3 {
                let scalar = f32::from_le_bytes(
                    bytes[i * 12 + c * 4..i * 12 + c * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
                let offset = count as usize * 12 + i * 64 + 48 + c * 4;
                let matrix = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                assert!(
                    (scalar - matrix).abs() < 1e-4 * (1.0 + scalar.abs()),
                    "scalar {scalar}, matrix {matrix}"
                );
                if scalar > 0.0 {
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "no matrix predictions tested");
    }
    #[cfg(test)]
    pub fn caustic_probe_means(&self, r: &super::Renderer) -> [f64; 2] {
        let count = 16384u32;
        let shader=r.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:Some("SPPC caustic correction probe"),
            source:wgpu::ShaderSource::Wgsl(super::ray_query::shader(&r.device,format!("{}\n{}",source(self.queries,self.hash_parameters.trailing_zeros()-5),r#"
@compute @workgroup_size(64)
fn sp_probe_caustic(@builtin(global_invocation_id) id:vec3<u32>){
 neural_pending=false;
 let value=trace(vec3<f32>(0.0,0.03,0.0),vec3<f32>(0.0,-1.0,0.0),hash(id.x^0x7a213f9u),0u);
 var color=value.radiance;
 if PHOTON_NRC_ENABLED && neural_pending {color+=neural_pending_scale*sp_predict(neural_pending_sample);}
 sp.gradients[id.x]=color.x;
}
"#)).into()),
        });
        let extra = self.resolve.get_bind_group_layout(1);
        let layout = r
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&r.layout), Some(&extra)],
                immediate_size: 0,
            });
        let stage = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(count) * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = r.device.create_command_encoder(&Default::default());
        for cached in 0..2u32 {
            let pipeline = r
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some("sp_probe_caustic"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[
                            ("PHOTON_NRC_ENABLED", f64::from(cached)),
                            ("ACTIVE_MODE", 0.),
                        ],
                        ..Default::default()
                    },
                    cache: None,
                });
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &r.bind_group, &[]);
                pass.set_bind_group(1, &self.group, &[]);
                pass.dispatch_workgroups(count / 64, 1, 1);
            }
            encoder.copy_buffer_to_buffer(
                &self.data,
                16 + 8515 * 16,
                &stage,
                u64::from(cached * count) * 4,
                u64::from(count) * 4,
            );
        }
        let bytes = r.read_buffer(encoder, &stage).unwrap();
        std::array::from_fn(|i| {
            bytes[i * count as usize * 4..(i + 1) * count as usize * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b) as f64)
                .sum::<f64>()
                / f64::from(count)
        })
    }
    pub fn matches(&self, s: &Settings) -> bool {
        self.queries == s.photon_nrc_queries.clamp(1024, 32768).next_power_of_two()
            && self.hash_parameters == (32u32 << s.photon_nrc_hash_log2.clamp(10, 19))
    }
    #[cfg(test)]
    pub fn query_offset(&self) -> u64 {
        4_495_936 + u64::from(self.hash_parameters) * 20
    }
    pub fn reset(&mut self) {
        self.initialized = false;
        self.batches = 0;
    }
    pub fn prepare(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        base: &wgpu::BindGroup,
        worlds: &[GpuWorld; 2],
        settings: &Settings,
    ) {
        let mut low = glam::Vec3::splat(-8.);
        let mut high = glam::Vec3::splat(8.);
        for w in worlds {
            low = low.min(w.encoding_bounds.0);
            high = high.max(w.encoding_bounds.1);
        }
        let extent = (high - low).max(glam::Vec3::ONE);
        low -= extent * 0.1;
        let scale = (extent * 1.2).recip();
        let p = Parameters {
            config: [
                settings.photon_nrc_photons.clamp(256, 524288),
                self.queries,
                self.batches,
                self.queries.div_ceil(50),
            ],
            radii: [
                settings.photon_nrc_global_radius.clamp(0.01, 2.),
                settings.photon_nrc_caustic_radius.clamp(0.005, 2.),
                0.3,
                2. / 3.,
            ],
            low: low.extend(0.).to_array(),
            scale: scale.extend(0.).to_array(),
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&p));
        if !self.initialized {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.init);
            pass.set_bind_group(0, base, &[]);
            pass.set_bind_group(1, &self.group, &[]);
            pass.dispatch_workgroups(
                1024,
                self.hash_parameters.max(self.queries).div_ceil(65536),
                1,
            );
            self.initialized = true;
        }
    }
    pub fn trace(&self, mode: usize, guides: bool) -> &wgpu::ComputePipeline {
        &self.trace[mode][usize::from(guides)]
    }
    pub fn resolve(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        base: &wgpu::BindGroup,
        size: [u32; 2],
        mode: usize,
        guides: bool,
        use_tensor: bool,
    ) {
        if use_tensor && let Some(tensor) = &self.tensor {
            tensor.run(
                encoder,
                base,
                &self.group,
                size[0] * size[1] * if mode == 2 { 2 } else { 1 },
            );
            if guides {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.guides);
                pass.set_bind_group(0, base, &[]);
                pass.set_bind_group(1, &self.group, &[]);
                pass.dispatch_workgroups((size[0] * size[1]).div_ceil(64), 1, 1);
            }
            return;
        }
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_bind_group(0, base, &[]);
        pass.set_bind_group(1, &self.group, &[]);
        pass.set_pipeline(&self.resolve);
        let groups = (size[0] * size[1] * if mode == 2 { 2 } else { 1 })
            .div_ceil(if self.cooperative { 16 } else { 64 });
        // Cooperative batches can exceed the per-axis dispatch limit at large
        // display sizes. The matrix shader linearizes this two-dimensional grid.
        pass.dispatch_workgroups(groups.min(65535), groups.div_ceil(65535), 1);
        if guides {
            pass.set_pipeline(&self.guides);
            pass.dispatch_workgroups((size[0] * size[1]).div_ceil(64), 1, 1);
        }
    }
    pub fn learn(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        base: &wgpu::BindGroup,
        photons: u32,
    ) {
        for (pipeline, x, y) in &self.passes {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, base, &[]);
            pass.set_bind_group(1, &self.group, &[]);
            pass.dispatch_workgroups(
                if *x == 0 {
                    photons.clamp(256, 524288).div_ceil(64)
                } else {
                    *x
                },
                *y,
                1,
            );
        }
        self.batches = self.batches.wrapping_add(1);
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn photon_nrc_shader_validates() {
        let text = super::source(8192, 14);
        let m = naga::front::wgsl::parse_str(&text)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&m)
        .unwrap();
        let span = m
            .types
            .iter()
            .find_map(|(_, t)| match (&t.name, &t.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == "SpStorage" => Some(*span),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            u64::from(span),
            4_495_936 + u64::from(32u32 << 14) * 20 + 8192 * 172 + 8192 / 64 + 8192
        );
    }
}
