//! GPU-only budget allocation, scans, compact active queues and indirect tracing.
use super::{Pixel, caustics::Caustics};
pub(super) struct Adaptive {
    pub indirect: wgpu::Buffer,
    reset: wgpu::ComputePipeline,
    scores: wgpu::ComputePipeline,
    weights: wgpu::ComputePipeline,
    allocate: wgpu::ComputePipeline,
    active: wgpu::ComputePipeline,
    compact: wgpu::ComputePipeline,
    guides: wgpu::ComputePipeline,
    traces: [Option<wgpu::ComputePipeline>; 12],
}
impl Adaptive {
    pub const INDIRECT_OFFSET: u64 = std::mem::offset_of!(Pixel, work) as u64 + 4;
    pub fn new(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
    ) -> Self {
        let pipeline = |entry: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let indirect = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Adaptive indirect dispatch"),
            size: 12,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            indirect,
            reset: pipeline("adaptive_reset"),
            scores: pipeline("adaptive_scores"),
            weights: pipeline("adaptive_weight_blocks"),
            allocate: pipeline("adaptive_allocate"),
            active: pipeline("adaptive_active_blocks"),
            compact: pipeline("adaptive_compact"),
            guides: pipeline("adaptive_guides"),
            traces: std::array::from_fn(|_| None),
        }
    }
    pub fn prepare(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        base: &wgpu::BindGroup,
        pixels: &wgpu::Buffer,
        size: [u32; 2],
        budget: [u32; 2],
        caustics: Option<&Caustics>,
    ) {
        let [samples, batch] = budget;
        let n = size[0] * size[1];
        if samples == 0 {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.reset);
            pass.set_bind_group(0, base, &[]);
            pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
        }
        if batch == 0 {
            encoder.clear_buffer(&self.indirect, 0, None);
            return;
        }
        for (pipeline, groups, photons) in [
            (&self.scores, n.div_ceil(256), None),
            (&self.weights, 1, None),
            (
                caustics.map_or(&self.allocate, |c| c.adaptive_allocate()),
                n.div_ceil(256),
                caustics,
            ),
            (&self.active, 1, None),
            (&self.compact, n.div_ceil(64), None),
        ] {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, base, &[]);
            if let Some(c) = photons {
                pass.set_bind_group(1, &c.group, &[]);
            }
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(pixels, Self::INDIRECT_OFFSET, &self.indirect, 0, 12);
    }
    pub fn trace_pipeline(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        mode: usize,
        neural: bool,
    ) -> &wgpu::ComputePipeline {
        self.traces[mode * 2 + usize::from(neural)].get_or_insert_with(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compact adaptive camera work"),
                layout: Some(layout),
                module: shader,
                entry_point: Some("accumulate_adaptive"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("ACTIVE_MODE", mode as f64),
                        ("ADAPTIVE_TRACE", 1.),
                        ("NRC_ENABLED", u32::from(neural) as f64),
                    ],
                    ..Default::default()
                },
                cache: None,
            })
        })
    }
    pub fn guides(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        base: &wgpu::BindGroup,
        size: [u32; 2],
    ) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.guides);
        pass.set_bind_group(0, base, &[]);
        pass.dispatch_workgroups(size[0].div_ceil(8), size[1].div_ceil(8), 1);
    }
}
