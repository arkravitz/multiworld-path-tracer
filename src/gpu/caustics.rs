//! Per-dispatch light paths, a collision-safe spatial hash and camera splats.
use super::{GpuWorld, buffer_entry};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Parameters {
    targets: [[f32; 4]; 32],
    layout: [u32; 4],
    shape: [f32; 4],
    spheres: [[f32; 4]; 16],
    learning: [u32; 4],
    features: [u32; 4],
}
pub(super) struct Caustics {
    pub group: wgpu::BindGroup,
    #[cfg(test)]
    pub data: wgpu::Buffer,
    #[cfg(test)]
    pub diagnostic_offset: u64,
    offsets: [u32; 4],
    uniform: wgpu::Buffer,
    learning: [u32; 4],
    round: u32,
    learn: wgpu::ComputePipeline,
    focus: wgpu::ComputePipeline,
    probabilities: wgpu::ComputePipeline,
    clear: wgpu::ComputePipeline,
    emit: [wgpu::ComputePipeline; 8],
    scan_buckets: wgpu::ComputePipeline,
    scan_blocks: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    adaptive_allocate: wgpu::ComputePipeline,
    adaptive_trace: [wgpu::ComputePipeline; 3],
    trace: [[wgpu::ComputePipeline; 2]; 3],
}
impl Caustics {
    pub fn new(
        device: &wgpu::Device,
        base: &wgpu::BindGroupLayout,
        width: u32,
        height: u32,
        trace_group_y: u32,
    ) -> Self {
        // Fixed capacity permits changing the independent launch budget without
        // recompiling pipelines. Only the requested lanes perform light tracing.
        let photons = 32768;
        let buckets = photons;
        let records = buckets * 4 + 256;
        let splats = records + photons * 2 * 17;
        let offsets = [photons, buckets, records, splats];
        let weights = splats + width * height * 6 + 8;
        let stats = weights + 1024 * 2 + 64;
        let focus = stats + 2048;
        let reservoirs = focus + 16;
        let learning = [weights, stats, focus, reservoirs];
        let data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Caustic photons, hash heads and pixel splats"),
            size: u64::from(reservoirs + photons * 2 * 2 * 20) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Progressive caustic parameters"),
            size: size_of::<Parameters>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bindings = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Caustic light transport"),
            entries: &[
                buffer_entry(0, wgpu::BufferBindingType::Storage { read_only: false }),
                buffer_entry(1, wgpu::BufferBindingType::Uniform),
            ],
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Caustic light transport"),
            layout: &bindings,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Light tracing and photon merging"),
            bind_group_layouts: &[Some(base), Some(&bindings)],
            immediate_size: 0,
        });
        let source = [
            concat!(
                include_str!("../shaders/trace.wgsl"),
                "\n",
                include_str!("../shaders/adaptive.wgsl")
            ),
            include_str!("../shaders/nrc.wgsl"),
            include_str!("../shaders/nrc_scalar.wgsl"),
            include_str!("../shaders/caustics.wgsl"),
            include_str!("../shaders/caustics_sampling.wgsl"),
            include_str!("../shaders/denoise.wgsl"),
        ]
        .join("\n");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Targeted caustic light paths"),
            source: wgpu::ShaderSource::Wgsl(super::ray_query::shader(device, source).into()),
        });
        let pipeline = |entry: &'static str, mode: usize, guides: bool| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("ACTIVE_MODE", mode.min(2) as f64),
                        ("TRACE_GROUP_Y", trace_group_y as f64),
                        (
                            "PHOTON_METHODS",
                            if entry == "photon_trace" {
                                mode as f64
                            } else {
                                0.
                            },
                        ),
                        (
                            "ADAPTIVE_TRACE",
                            u32::from(entry == "accumulate_adaptive") as f64,
                        ),
                        ("DENOISE_INPUT", u32::from(guides) as f64),
                    ],
                    ..Default::default()
                },
                cache: None,
            })
        };
        Self {
            group,
            #[cfg(test)]
            data,
            #[cfg(test)]
            diagnostic_offset: u64::from(weights - 8) * 4,
            offsets,
            uniform,
            learning,
            round: 0,
            learn: pipeline("photon_learn", 0, false),
            focus: pipeline("photon_focus", 0, false),
            probabilities: pipeline("photon_probabilities", 0, false),
            clear: pipeline("photon_clear", 0, false),
            emit: std::array::from_fn(|methods| pipeline("photon_trace", methods, false)),
            scan_buckets: pipeline("photon_scan_buckets", 0, false),
            scan_blocks: pipeline("photon_scan_blocks", 0, false),
            scatter: pipeline("photon_scatter", 0, false),
            adaptive_allocate: pipeline("adaptive_allocate", 0, false),
            adaptive_trace: std::array::from_fn(|m| pipeline("accumulate_adaptive", m, false)),
            trace: std::array::from_fn(|m| {
                std::array::from_fn(|g| pipeline("accumulate", m, g != 0))
            }),
        }
    }
    pub fn prepare(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene: &wgpu::BindGroup,
        worlds: &[GpuWorld; 2],
        frame: [u32; 3],
        settings: &crate::settings::Settings,
    ) {
        let [width, height, samples] = frame;
        let [photons, buckets, _, _] = self.offsets;
        // Accumulated photon count times r^2 grows while r tends to zero.
        let radius = 0.16 * (1.0 + samples as f32 / 64.0).powf(-1.0 / 6.0);
        if samples == 0 {
            self.round = 0;
        }
        let active = if settings.photon_budget == 0 {
            (width * height / 4)
                .max(1)
                .next_power_of_two()
                .clamp(4096, photons)
        } else {
            settings.photon_budget.clamp(256, photons)
        };
        let parameters = Parameters {
            targets: std::array::from_fn(|i| {
                worlds[i / 16]
                    .glass_targets
                    .get(i % 16)
                    .copied()
                    .unwrap_or([0.; 4])
            }),
            layout: self.offsets,
            spheres: std::array::from_fn(|i| {
                worlds[i / 8]
                    .glass_spheres
                    .get((i % 8) / 2)
                    .map_or([0.; 4], |s| s[i % 2])
            }),
            learning: self.learning,
            features: [
                u32::from(settings.photon_guiding)
                    | (u32::from(settings.manifold_guiding) << 1)
                    | (u32::from(settings.photon_reuse) << 2),
                active,
                self.round,
                worlds[0].glass_spheres.len() as u32
                    | ((worlds[1].glass_spheres.len() as u32) << 8),
            ],
            shape: [
                radius,
                worlds[0].glass_targets.len() as f32,
                worlds[1].glass_targets.len() as f32,
                0.,
            ],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&parameters));
        // Separate passes establish global storage visibility between clearing,
        // sorting and gathering photons. No per-bucket cap drops energy.
        for (pipeline, groups, rows) in [
            (&self.focus, 2, 1),
            (&self.learn, 16, 1),
            (&self.probabilities, 1, 1),
            (
                &self.clear,
                1024,
                (width * height * 6).max(photons * 2).div_ceil(65536),
            ),
            (
                &self.emit[parameters.features[0] as usize],
                active.div_ceil(64),
                2,
            ),
            (&self.scan_buckets, (buckets * 2) / 256, 1),
            (&self.scan_blocks, 1, 1),
            (&self.scatter, (photons * 2).div_ceil(64), 1),
        ] {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, scene, &[]);
            pass.set_bind_group(1, &self.group, &[]);
            pass.dispatch_workgroups(groups, rows, 1);
        }
        self.round += 1;
    }
    pub fn adaptive_allocate(&self) -> &wgpu::ComputePipeline {
        &self.adaptive_allocate
    }
    pub fn adaptive_trace(&self, mode: usize) -> &wgpu::ComputePipeline {
        &self.adaptive_trace[mode]
    }
    pub fn trace_pipeline(&self, mode: usize, guides: bool) -> &wgpu::ComputePipeline {
        &self.trace[mode][usize::from(guides)]
    }
}

/// Each analytic caster gets a bound; consecutive mesh triangles share one.
/// Grouping only changes the launch proposal, never geometry or transport.
fn merge_bound(a: &mut (glam::Vec3, glam::Vec3), b: (glam::Vec3, glam::Vec3)) {
    a.0 = a.0.min(b.0);
    a.1 = a.1.max(b.1);
}
pub(super) fn targets(objects: &[crate::scene::Object], bvh: &crate::bvh::Bvh) -> Vec<[f32; 4]> {
    let mut bounds = vec![None; objects.len()];
    for node in &bvh.nodes {
        if node.object != crate::bvh::INTERIOR {
            bounds[node.object as usize] = Some((
                glam::Vec3::from_array(node.low),
                glam::Vec3::from_array(node.high),
            ));
        }
    }
    let mut groups: Vec<(glam::Vec3, glam::Vec3)> = Vec::new();
    let mut exact_radii = Vec::new();
    let mut previous_mesh = false;
    for (o, bound) in objects.iter().zip(bounds) {
        if o.color_material[3] < 2. {
            previous_mesh = false;
            continue;
        }
        let Some(bound) = bound else {
            continue;
        };
        let mesh = o.extent_shape[3] == 5.;
        if (mesh && previous_mesh) || groups.len() == 16 {
            merge_bound(groups.last_mut().unwrap(), bound);
            *exact_radii.last_mut().unwrap() = None;
        } else {
            groups.push(bound);
            exact_radii.push(match o.extent_shape[3] {
                0. => Some(o.center_radius[3]),
                4. => Some(o.extent_shape[..3].iter().copied().fold(0_f32, f32::max)),
                _ => None,
            });
        }
        previous_mesh = mesh;
    }
    groups
        .into_iter()
        .zip(exact_radii)
        .map(|((low, high), radius)| {
            ((low + high) * 0.5)
                .extend(radius.unwrap_or((high - low).length() * 0.5) * 1.0002)
                .to_array()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn caustic_shader_validates() {
        let source = [
            concat!(
                include_str!("../shaders/trace.wgsl"),
                "\n",
                include_str!("../shaders/adaptive.wgsl")
            ),
            include_str!("../shaders/nrc.wgsl"),
            include_str!("../shaders/nrc_scalar.wgsl"),
            include_str!("../shaders/caustics.wgsl"),
            include_str!("../shaders/caustics_sampling.wgsl"),
            include_str!("../shaders/denoise.wgsl"),
        ]
        .join("\n");
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let span = module
            .types
            .iter()
            .find_map(|(_, ty)| {
                if ty.name.as_deref() == Some("CausticParams")
                    && let naga::TypeInner::Struct { span, .. } = ty.inner
                {
                    return Some(span);
                }
                None
            })
            .unwrap();
        assert_eq!(span as usize, std::mem::size_of::<super::Parameters>());
    }
}
