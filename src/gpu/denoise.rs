use super::*;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct Frame {
    pub sample: [f32; 4],
    pub position_a: [f32; 4],
    pub normal_a: [f32; 4],
    pub albedo_a: [f32; 4],
    pub position_b: [f32; 4],
    pub normal_b: [f32; 4],
    pub albedo_b: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FilterParams {
    camera: [[f32; 4]; 4],
    flags: [u32; 4],
    controls: [f32; 4],
}
pub(super) struct Denoiser {
    pub trace: [wgpu::ComputePipeline; 6],
    pipeline: wgpu::ComputePipeline,
    uniform: wgpu::Buffer,
    groups: [wgpu::BindGroup; 2],
    #[allow(dead_code)] // Keeps history accessible to GPU regression tests.
    pub history: [wgpu::Buffer; 2],
    pub current: usize,
    valid: bool,
    previous: Option<Settings>,
}
impl Denoiser {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        shader: &wgpu::ShaderModule,
        base: &wgpu::BindGroupLayout,
        trace_layout: &wgpu::PipelineLayout,
    ) -> Self {
        let trace = std::array::from_fn(|mode| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Denoiser input trace"),
                layout: Some(trace_layout),
                module: shader,
                entry_point: Some("accumulate"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("ACTIVE_MODE", mode as f64), ("DENOISE_INPUT", 1.)],
                    ..Default::default()
                },
                cache: None,
            })
        });
        let history = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dual-world denoising history"),
                size: u64::from(width) * u64::from(height) * size_of::<Frame>() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temporal reprojection parameters"),
            size: size_of::<FilterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Temporal histories"),
            entries: &[
                buffer_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                buffer_entry(2, wgpu::BufferBindingType::Uniform),
            ],
        });
        let groups = std::array::from_fn(|read| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Ping-pong temporal history"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: history[read].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: history[1 - read].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            })
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Display-only denoiser"),
            bind_group_layouts: &[Some(base), Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dual-world temporal denoiser"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: Some("denoise"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            trace,
            pipeline,
            uniform,
            groups,
            history,
            current: 0,
            valid: false,
            previous: None,
        }
    }
    pub fn reset(&mut self) {
        self.valid = false;
    }
    pub fn encode(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        base: &wgpu::BindGroup,
        settings: &Settings,
        width: u32,
        height: u32,
    ) {
        let compatible = self.valid
            && self
                .previous
                .as_ref()
                .is_some_and(|old| settings.history_compatible(old));
        let moved = self.previous.as_ref().is_some_and(|old| {
            old.yaw != settings.yaw
                || old.pitch != settings.pitch
                || old.distance != settings.distance
        });
        let camera = self
            .previous
            .as_ref()
            .unwrap_or(settings)
            .camera(width as f32 / height as f32);
        let params = FilterParams {
            camera,
            flags: [compatible as u32, moved as u32, 0, 0],
            controls: [settings.denoise_strength, 32., 0., 0.],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&params));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Temporal reprojection and guide filter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, base, &[]);
            pass.set_bind_group(1, &self.groups[self.current], &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        self.current = 1 - self.current;
        self.valid = true;
        self.previous = Some(settings.clone());
    }
}
