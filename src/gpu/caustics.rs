//! Per-dispatch light paths, a collision-safe spatial hash and camera splats.
use super::{GpuWorld, buffer_entry};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Parameters {
    targets: [[f32; 4]; 32],
    layout: [u32; 4],
    shape: [f32; 4],
}
pub(super) struct Caustics {
    pub group: wgpu::BindGroup,
    offsets: [u32; 4],
    uniform: wgpu::Buffer,
    clear: wgpu::ComputePipeline,
    emit: wgpu::ComputePipeline,
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
    ) -> Self {
        // Small previews should not spend more work emitting photons than
        // tracing camera pixels. Scale the budget and cap interactive memory.
        let photons = (width * height / 4)
            .max(1)
            .next_power_of_two()
            .clamp(4096, 32768);
        let buckets = photons;
        let records = buckets * 4 + 256;
        let splats = records + photons * 2 * 13;
        let offsets = [photons, buckets, records, splats];
        let data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Caustic photons, hash heads and pixel splats"),
            size: (u64::from(splats) + u64::from(width) * u64::from(height) * 6) * 4,
            usage: wgpu::BufferUsages::STORAGE,
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
            include_str!("../shaders/denoise.wgsl"),
        ]
        .join("\n");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Targeted caustic light paths"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = |entry: &'static str, mode: usize, guides: bool| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("ACTIVE_MODE", mode as f64),
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
            offsets,
            uniform,
            clear: pipeline("photon_clear", 0, false),
            emit: pipeline("photon_trace", 0, false),
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
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene: &wgpu::BindGroup,
        worlds: &[GpuWorld; 2],
        size: [u32; 2],
        samples: u32,
    ) {
        let [width, height] = size;
        let [photons, buckets, _, _] = self.offsets;
        // Accumulated photon count times r^2 grows while r tends to zero.
        let radius = 0.16 * (1.0 + samples as f32 / 64.0).powf(-1.0 / 6.0);
        let parameters = Parameters {
            targets: std::array::from_fn(|i| {
                worlds[i / 16]
                    .glass_targets
                    .get(i % 16)
                    .copied()
                    .unwrap_or([0.; 4])
            }),
            layout: self.offsets,
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
            (
                &self.clear,
                1024,
                (width * height * 6).max(buckets * 2).div_ceil(65536),
            ),
            (&self.emit, photons.div_ceil(64), 2),
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
    }
}
