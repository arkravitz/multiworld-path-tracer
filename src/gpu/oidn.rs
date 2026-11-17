//! Asynchronous Metal OIDN. Bounded readback/job queues and generation stamps
//! prevent stale camera/scene results from becoming visible. No worker waits in
//! the interactive render loop; explicit exports can request a finished image.
mod native;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use std::{
    sync::mpsc,
    time::{Duration, Instant},
};
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Input {
    pub color_a: [f32; 4],
    pub normal_a: [f32; 4],
    pub albedo_a: [f32; 4],
    pub color_b: [f32; 4],
    pub normal_b: [f32; 4],
    pub albedo_b: [f32; 4],
    pub half_a0: [f32; 4],
    pub half_a1: [f32; 4],
    pub half_b0: [f32; 4],
    pub half_b1: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct Feedback {
    pub color_a: [f32; 4],
    pub raw_a: [f32; 4],
    pub color_b: [f32; 4],
    pub raw_b: [f32; 4],
}
struct Job {
    samples: u32,
    generation: u64,
    size: [u32; 2],
    blend: bool,
    input: wgpu::Buffer,
    uncertainty: bool,
    strength: f32,
}
struct Output {
    generation: u64,
    samples: u32,
    image: Vec<Feedback>,
    milliseconds: f64,
    prior: bool,
    shared: bool,
}
struct Pending {
    samples: u32,
    buffer: wgpu::Buffer,
    mapped: bool,
    receive: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    send: Option<mpsc::SyncSender<Result<(), wgpu::BufferAsyncError>>>,
    generation: u64,
    blend: bool,
    uncertainty: bool,
    strength: f32,
}
pub(super) struct Snapshot {
    pub generation: u64,
    pub samples: u32,
    pub blend: bool,
    pub force: bool,
    pub uncertainty: bool,
    pub strength: f32,
}
pub(super) struct Oidn {
    jobs: mpsc::SyncSender<Job>,
    outputs: mpsc::Receiver<Result<Output, String>>,
    pending: Option<Pending>,
    busy: bool,
    input: wgpu::Buffer,
    shared: bool,
    finalize: wgpu::ComputePipeline,
    finalize_group: wgpu::BindGroup,
    finalize_uniform: wgpu::Buffer,
    snapshot_params: [u32; 8],
    snapshot_group: Option<wgpu::BindGroup>,
    output: wgpu::Buffer,
    input_group: wgpu::BindGroup,
    output_group: wgpu::BindGroup,
    pack: wgpu::ComputePipeline,
    publish: wgpu::ComputePipeline,
    feedback_dirty: bool,
    present: wgpu::ComputePipeline,
    size: [u32; 2],
    last_request: u32,
    completed: u32,
    last_started: Instant,
    last_prior: Instant,
    pub status: String,
    pub failed: bool,
}
impl Oidn {
    pub fn new(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        base: &wgpu::BindGroupLayout,
        size: [u32; 2],
    ) -> Self {
        let layout = |binding| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("OIDN packed image"),
                entries: &[super::buffer_entry(
                    binding,
                    wgpu::BufferBindingType::Storage { read_only: false },
                )],
            })
        };
        let input_layout = layout(4);
        let output_layout = layout(5);
        let n = u64::from(size[0]) * u64::from(size[1]);
        let buffer = |label, bytes| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let shared = if std::env::var_os("MULTIWORLD_OIDN_STAGING").is_none() {
            super::metal::SharedBuffer::new(device, n * size_of::<Input>() as u64 * 2).ok()
        } else {
            None
        };
        let input = shared.as_ref().map_or_else(
            || buffer("OIDN input packing", n * size_of::<Input>() as u64),
            |s| s.buffer.clone(),
        );
        let output = buffer(
            "OIDN completed reconstruction",
            n * size_of::<Feedback>() as u64,
        );
        let group = |buffer: &wgpu::Buffer, layout: &wgpu::BindGroupLayout, binding| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("OIDN image"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding,
                    resource: buffer.as_entire_binding(),
                }],
            })
        };
        let finalize_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("OIDN snapshot parameters"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let finalize_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shared OIDN publication"),
            entries: &[
                super::buffer_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
                super::buffer_entry(5, wgpu::BufferBindingType::Storage { read_only: false }),
                super::buffer_entry(6, wgpu::BufferBindingType::Uniform),
            ],
        });
        let finalize_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shared OIDN publication"),
            layout: &finalize_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: finalize_uniform.as_entire_binding(),
                },
            ],
        });
        let input_group = group(&input, &input_layout, 4);
        let output_group = group(&output, &output_layout, 5);
        let pipeline = |entry: &'static str, extra: &wgpu::BindGroupLayout| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &[Some(base), Some(extra)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (jobs, work) = mpsc::sync_channel::<Job>(1);
        let (results, outputs) = mpsc::sync_channel(1);
        let use_shared = shared.is_some();
        std::thread::Builder::new()
            .name("OIDN reconstruction".into())
            .spawn(move || {
                let mut engine =
                    match native::Engine::new_on_queue(shared.as_ref().map(|s| s.native_queue())) {
                        Ok(e) => e,
                        Err(e) => {
                            let _ = results.send(Err(format!("{e:#}")));
                            return;
                        }
                    };
                while let Ok(job) = work.recv() {
                    let started = Instant::now();
                    let result = (|| -> Result<Output> {
                        if let Some(shared) = &shared {
                            engine.reconstruct_shared(
                                job.size,
                                shared.native_buffer(),
                                job.blend,
                                job.uncertainty,
                            )?;
                            return Ok(Output {
                                generation: job.generation,
                                samples: job.samples,
                                image: Vec::new(),
                                milliseconds: started.elapsed().as_secs_f64() * 1000.,
                                prior: job.uncertainty,
                                shared: true,
                            });
                        }
                        // Copy mapped data on this worker, not the UI thread.
                        let bytes = job.input.slice(..).get_mapped_range()?;
                        let input: Vec<Input> = bytes
                            .as_chunks::<{ size_of::<Input>() }>()
                            .0
                            .iter()
                            .map(|p| bytemuck::pod_read_unaligned(p))
                            .collect();
                        drop(bytes);
                        job.input.unmap();
                        let a = engine.reconstruct(job.size, &input, 0, job.uncertainty)?;
                        let b = if job.blend {
                            Some(engine.reconstruct(job.size, &input, 1, job.uncertainty)?)
                        } else {
                            None
                        };
                        let samples = input.first().map_or(0, |p| p.color_a[3] as u32);
                        let variance =
                            |i: usize, world: usize, reconstruction: &native::Reconstruction| {
                                let Some(halves) = &reconstruction.halves else {
                                    return -1.0;
                                };
                                let (even, odd) = if world == 0 {
                                    (input[i].half_a0, input[i].half_a1)
                                } else {
                                    (input[i].half_b0, input[i].half_b1)
                                };
                                let n = even[3] + odd[3];
                                if n <= 1.0 {
                                    return -1.0;
                                }
                                let ratio = even[3] * odd[3] / (n * n);
                                let mut variance = 0.0;
                                for (c, weight) in [0.2126, 0.7152, 0.0722].into_iter().enumerate()
                                {
                                    let x = even[c] + job.strength * (halves[0][i][c] - even[c]);
                                    let y = odd[c] + job.strength * (halves[1][i][c] - odd[c]);
                                    variance += weight * (x - y) * (x - y);
                                }
                                // Store per-ray equivalent variance, so the prior
                                // doesn't increase merely because its snapshot ages.
                                (ratio * variance * n).max(0.0)
                            };
                        let mut image = Vec::with_capacity(input.len());
                        for (i, p) in input.iter().enumerate() {
                            let mut pixel = Feedback {
                                color_a: p.color_a,
                                raw_a: p.color_a,
                                color_b: p.color_b,
                                raw_b: p.color_b,
                            };
                            pixel.color_a[..3].copy_from_slice(&a.color[i]);
                            pixel.raw_a[3] = variance(i, 0, &a);
                            pixel.raw_b[3] = -1.0;
                            if let Some(b) = &b {
                                pixel.color_b[..3].copy_from_slice(&b.color[i]);
                                pixel.raw_b[3] = variance(i, 1, b);
                            }
                            image.push(pixel);
                        }
                        Ok(Output {
                            generation: job.generation,
                            samples,
                            image,
                            milliseconds: started.elapsed().as_secs_f64() * 1000.,
                            prior: job.uncertainty,
                            shared: false,
                        })
                    })();
                    if results.send(result.map_err(|e| format!("{e:#}"))).is_err() {
                        break;
                    }
                }
            })
            .expect("create OIDN worker");
        Self {
            jobs,
            outputs,
            pending: None,
            busy: false,
            input,
            shared: use_shared,
            finalize: pipeline("oidn_finalize", &finalize_layout),
            finalize_group,
            finalize_uniform,
            snapshot_params: [0; 8],
            snapshot_group: None,
            output,
            input_group,
            output_group,
            pack: pipeline("oidn_pack", &input_layout),
            present: pipeline("oidn_present", &output_layout),
            publish: pipeline("oidn_publish", &output_layout),
            feedback_dirty: false,
            size,
            last_request: 0,
            completed: 0,
            last_started: Instant::now() - Duration::from_secs(1),
            last_prior: Instant::now() - Duration::from_secs(10),
            status: "OIDN / Metal · preparing".into(),
            failed: false,
        }
    }
    pub fn reset(&mut self) {
        self.completed = 0;
        self.last_request = 0;
        self.last_prior = Instant::now() - Duration::from_secs(10);
        self.feedback_dirty = false;
    }
    pub fn collect(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, generation: u64) {
        let _ = device.poll(wgpu::PollType::Poll);
        let mapped_result = self
            .pending
            .as_ref()
            .filter(|pending| pending.mapped)
            .and_then(|pending| pending.receive.try_recv().ok());
        if let Some(result) = mapped_result {
            let pending = self.pending.take().unwrap();
            match result {
                Ok(()) => {
                    if pending.generation == generation {
                        let job = Job {
                            generation,
                            samples: pending.samples,
                            size: self.size,
                            blend: pending.blend,
                            input: pending.buffer,
                            uncertainty: pending.uncertainty,
                            strength: pending.strength,
                        };
                        match self.jobs.try_send(job) {
                            Ok(()) => self.busy = true,
                            Err(
                                mpsc::TrySendError::Full(job)
                                | mpsc::TrySendError::Disconnected(job),
                            ) => {
                                if !self.shared {
                                    job.input.unmap();
                                }
                                self.failed = true;
                                self.status = "OIDN worker stopped".into();
                            }
                        }
                    } else {
                        if !self.shared {
                            pending.buffer.unmap();
                        }
                    }
                }
                Err(error) => {
                    self.failed = true;
                    self.status = format!("OIDN readback: {error}");
                }
            }
        }
        while let Ok(result) = self.outputs.try_recv() {
            self.busy = false;
            match result {
                Ok(output) if output.generation == generation => {
                    if output.shared {
                        queue.write_buffer(
                            &self.finalize_uniform,
                            0,
                            bytemuck::cast_slice(&self.snapshot_params),
                        );
                        let mut encoder = device.create_command_encoder(&Default::default());
                        {
                            let mut pass = encoder.begin_compute_pass(&Default::default());
                            pass.set_pipeline(&self.finalize);
                            pass.set_bind_group(0, self.snapshot_group.as_ref().unwrap(), &[]);
                            pass.set_bind_group(1, &self.finalize_group, &[]);
                            pass.dispatch_workgroups(
                                self.size[0].div_ceil(8),
                                self.size[1].div_ceil(8),
                                1,
                            );
                        }
                        queue.submit([encoder.finish()]);
                    } else {
                        queue.write_buffer(&self.output, 0, bytemuck::cast_slice(&output.image));
                    }
                    self.completed = output.samples;
                    self.feedback_dirty |= output.prior;
                    self.status = format!(
                        "OIDN / Metal · {} · {} spp · {:.0} ms",
                        if self.shared {
                            "shared buffers"
                        } else {
                            "staged"
                        },
                        output.samples,
                        output.milliseconds
                    );
                }
                Ok(_) => {}
                Err(error) => {
                    self.failed = true;
                    self.status = error;
                }
            }
        }
    }
    pub fn request(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        base: &wgpu::BindGroup,
        snapshot: Snapshot,
    ) {
        let Snapshot {
            generation,
            samples,
            blend,
            force,
            uncertainty,
            strength,
        } = snapshot;
        let uncertainty =
            uncertainty && (force || self.last_prior.elapsed() >= Duration::from_millis(750));
        if self.failed || samples == 0 || self.busy || self.pending.is_some() {
            return;
        }
        if !force
            && (samples < 4
                || samples < self.last_request + 8
                || self.last_started.elapsed() < Duration::from_millis(150))
        {
            return;
        }
        if samples == self.completed || samples == self.last_request {
            return;
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pack);
            pass.set_bind_group(0, base, &[]);
            pass.set_bind_group(1, &self.input_group, &[]);
            pass.dispatch_workgroups(self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1);
        }
        self.snapshot_group = Some(base.clone());
        self.snapshot_params = [
            self.size[0],
            self.size[1],
            u32::from(blend),
            u32::from(uncertainty),
            strength.to_bits(),
            0,
            0,
            0,
        ];
        let buffer = if self.shared {
            self.input.clone()
        } else {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Asynchronous OIDN readback"),
                size: self.input.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        if !self.shared {
            encoder.copy_buffer_to_buffer(&self.input, 0, &buffer, 0, self.input.size());
        }
        let (send, receive) = mpsc::sync_channel(1);
        self.pending = Some(Pending {
            samples,
            buffer,
            mapped: false,
            receive,
            send: Some(send),
            generation,
            blend,
            uncertainty,
            strength,
        });
        self.last_request = samples;
        self.last_started = Instant::now();
        if uncertainty {
            self.last_prior = Instant::now();
        }
    }
    pub fn after_submit(&mut self, queue: &wgpu::Queue) {
        if let Some(p) = &mut self.pending
            && !p.mapped
        {
            let send = p.send.take().unwrap();
            if self.shared {
                queue.on_submitted_work_done(move || {
                    let _ = send.send(Ok(()));
                });
            } else {
                p.buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    let _ = send.send(r);
                });
            }
            p.mapped = true;
        }
    }
    pub fn publish_feedback(&mut self, encoder: &mut wgpu::CommandEncoder, base: &wgpu::BindGroup) {
        if !self.feedback_dirty {
            return;
        }
        self.feedback_dirty = false;
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.publish);
        pass.set_bind_group(0, base, &[]);
        pass.set_bind_group(1, &self.output_group, &[]);
        pass.dispatch_workgroups(self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1);
    }
    pub fn present(&self, encoder: &mut wgpu::CommandEncoder, base: &wgpu::BindGroup) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.present);
        pass.set_bind_group(0, base, &[]);
        pass.set_bind_group(1, &self.output_group, &[]);
        pass.dispatch_workgroups(self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1);
    }
    pub fn output_buffer(&self) -> &wgpu::Buffer {
        &self.output
    }
    pub fn completed_samples(&self) -> u32 {
        self.completed
    }
    pub fn ready(&self, samples: u32) -> bool {
        self.completed == samples && samples > 0
    }
    pub fn has_image(&self) -> bool {
        self.completed > 0
    }
    pub fn has_pending_work(&self) -> bool {
        self.busy || self.pending.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires installed OIDN runtime and Metal GPU"]
    fn shared_metal_oidn_matches_staging_for_both_worlds_and_halves() {
        let (device, queue) = pollster::block_on(super::super::headless_device()).unwrap();
        let r = super::super::Renderer::new(device, queue, 32, 24, 42).unwrap();
        let size = [32, 24];
        let n = (size[0] * size[1]) as usize;
        let mut input = vec![Input::zeroed(); n];
        for (i, p) in input.iter_mut().enumerate() {
            let noise = ((i * 7919 % 101) as f32 / 100. - 0.5) * 0.2;
            let c = if i % 32 < 16 { 0.01 } else { 1.2 };
            p.color_a = [c + noise, 0.3, 0.7, 32.];
            p.color_b = [0.5, c + noise, 0.2, 32.];
            p.normal_a = [0., 1., 0., 0.];
            p.normal_b = p.normal_a;
            p.albedo_a = [c.min(1.), 0.3, 0.7, 0.];
            p.albedo_b = [0.5, c.min(1.), 0.2, 0.];
            p.half_a0 = [c + noise, 0.3, 0.7, 16.];
            p.half_a1 = [c - noise, 0.3, 0.7, 16.];
            p.half_b0 = [0.5, c + noise, 0.2, 16.];
            p.half_b1 = [0.5, c - noise, 0.2, 16.];
        }
        let bytes = std::mem::size_of_val(input.as_slice()) as u64;
        let shared = super::super::metal::SharedBuffer::new(&r.device, bytes * 2).unwrap();
        r.queue
            .write_buffer(&shared.buffer, 0, bytemuck::cast_slice(&input));
        r.queue.submit([]);
        r.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(60)),
            })
            .unwrap();
        let mut native = native::Engine::new_on_queue(Some(shared.native_queue())).unwrap();
        native
            .reconstruct_shared(size, shared.native_buffer(), true, true)
            .unwrap();
        let stage = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shared OIDN equivalence readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = r.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&shared.buffer, bytes, &stage, 0, bytes);
        let result = r.read_buffer(encoder, &stage).unwrap();
        let result: Vec<Input> = result
            .as_chunks::<{ size_of::<Input>() }>()
            .0
            .iter()
            .map(|p| bytemuck::pod_read_unaligned(p))
            .collect();
        let mut staged = native::Engine::new().unwrap();
        let mut max_error = 0_f32;
        for world in 0..2 {
            let expected = staged.reconstruct(size, &input, world, true).unwrap();
            for (i, p) in result.iter().enumerate() {
                let actual = if world == 0 {
                    [p.color_a, p.half_a0, p.half_a1]
                } else {
                    [p.color_b, p.half_b0, p.half_b1]
                };
                for (a, b) in actual.iter().zip([
                    expected.color[i],
                    expected.halves.as_ref().unwrap()[0][i],
                    expected.halves.as_ref().unwrap()[1][i],
                ]) {
                    for c in 0..3 {
                        max_error = max_error.max((a[c] - b[c]).abs());
                    }
                }
            }
        }
        eprintln!("Shared/staged OIDN maximum RGB error: {max_error}");
        assert!(
            max_error < 1e-4,
            "Shared Metal reconstruction differs from staged reconstruction"
        );
    }
    #[test]
    fn oidn_shader_and_pixel_abi_validate() {
        assert_eq!(size_of::<Input>(), 160);
        assert_eq!(size_of::<Feedback>(), 64);
        assert_eq!(size_of::<super::super::Pixel>(), 320);
        let source = [
            concat!(
                include_str!("../shaders/trace.wgsl"),
                "\n",
                include_str!("../shaders/adaptive.wgsl")
            ),
            include_str!("../shaders/nrc.wgsl"),
            include_str!("../shaders/nrc_scalar.wgsl"),
            include_str!("../shaders/caustics_stub.wgsl"),
            include_str!("../shaders/denoise.wgsl"),
            include_str!("../shaders/oidn.wgsl"),
        ]
        .join("\n");
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        for (name, bytes) in [
            ("Pixel", size_of::<super::super::Pixel>()),
            ("OidnInput", size_of::<Input>()),
            ("OidnFeedback", size_of::<Feedback>()),
        ] {
            let (_, ty) = module
                .types
                .iter()
                .find(|(_, ty)| ty.name.as_deref() == Some(name))
                .unwrap();
            let naga::TypeInner::Struct { span, members } = &ty.inner else {
                panic!("expected a struct")
            };
            assert_eq!(*span as usize, bytes, "{name} CPU/GPU layout mismatch");
            if name == "Pixel" {
                assert_eq!(
                    members
                        .iter()
                        .find(|m| m.name.as_deref() == Some("work"))
                        .unwrap()
                        .offset as usize,
                    std::mem::offset_of!(super::super::Pixel, work)
                );
            }
        }
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
    #[test]
    #[ignore = "requires installed OIDN runtime and a Metal GPU on macOS"]
    fn metal_oidn_preserves_edges_and_reduces_known_image_error() {
        let mut engine = native::Engine::new().unwrap();
        let size = [64, 48];
        let mut input = vec![Input::zeroed(); (size[0] * size[1]) as usize];
        let mut truth = Vec::new();
        for y in 0..size[1] {
            for x in 0..size[0] {
                let i = (y * size[0] + x) as usize;
                let color: [f32; 3] = if x < 32 {
                    [0.1, 0.3, 0.7]
                } else {
                    [1.2, 0.8, 0.2]
                };
                truth.push(color);
                for (c, value) in color.into_iter().enumerate() {
                    let mut bits = (i as u32 + 937u32 * c as u32 + 191)
                        .wrapping_mul(747796405)
                        .wrapping_add(2891336453);
                    bits = ((bits >> ((bits >> 28) + 4)) ^ bits).wrapping_mul(277803737);
                    let noise = ((bits >> 8) as f32 / 16777216. - 0.5) * 0.8;
                    input[i].color_a[c] = (value + noise).max(0.);
                    input[i].color_b[c] = (2. * value + noise).max(0.);
                    input[i].albedo_a[c] = value.min(1.);
                    input[i].albedo_b[c] = value.min(1.);
                }
                input[i].normal_a = [0., 1., 0., 0.];
                input[i].normal_b = [0., 1., 0., 0.];
            }
        }
        for world in 0..2 {
            let output = engine.filter(size, &input, world).unwrap();
            let mut raw = 0_f64;
            let mut clean = 0_f64;
            for (i, truth) in truth.iter().enumerate() {
                for c in 0..3 {
                    let target = truth[c] * (world as f32 + 1.);
                    let noisy = if world == 0 {
                        input[i].color_a[c]
                    } else {
                        input[i].color_b[c]
                    };
                    raw += (noisy - target).powi(2) as f64;
                    clean += (output[i][c] - target).powi(2) as f64;
                }
            }
            eprintln!(
                "OIDN Metal world {world}: raw MSE {}, filtered {}",
                raw / (input.len() * 3) as f64,
                clean / (input.len() * 3) as f64
            );
            assert!(clean < raw * 0.4, "OIDN did not improve this known signal");
            assert!(
                output[30][2] > output[33][2] + 0.2,
                "material edge was lost"
            );
        }
        assert!(engine.filter([2, 2], &input, 0).is_err());
    }
}
