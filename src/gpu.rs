mod adaptive;
mod caustics;
mod denoise;
mod nrc;
mod oidn;
use crate::{bvh::Bvh, scene, settings::Settings};
use anyhow::{Context, Result, ensure};
use bytemuck::{Pod, Zeroable};
use std::{fs::File, io::BufWriter, path::Path, sync::mpsc, time::Duration};
use wgpu::util::DeviceExt;

pub fn instance_descriptor() -> wgpu::InstanceDescriptor {
    wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    }
}
pub fn device_descriptor(adapter: &wgpu::Adapter) -> wgpu::DeviceDescriptor<'static> {
    let needed = wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
    let matrices = if adapter.get_info().backend == wgpu::Backend::Metal
        && adapter.features().contains(needed)
    {
        needed
    } else {
        wgpu::Features::empty()
    };
    wgpu::DeviceDescriptor {
        label: Some("Multiworld compute device"),
        required_features: matrices,
        // The optional hardware matrix path is explicitly feature-gated. The
        // fallback never parses experimental WGSL on unsupported adapters.
        experimental_features: if matrices.is_empty() {
            wgpu::ExperimentalFeatures::disabled()
        } else {
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        },
        required_limits: wgpu::Limits {
            max_storage_buffer_binding_size: adapter
                .limits()
                .max_storage_buffer_binding_size
                .min(512 * 1024 * 1024),
            max_buffer_size: adapter.limits().max_buffer_size.min(512 * 1024 * 1024),
            ..Default::default()
        },
        ..Default::default()
    }
}
pub async fn headless_device() -> Result<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(instance_descriptor());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .context("No compute-capable GPU adapter (Metal/Vulkan/DX12) found")?;
    let info = adapter.get_info();
    eprintln!("GPU: {} / {:?}", info.name, info.backend);
    Ok(adapter.request_device(&device_descriptor(&adapter)).await?)
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    size: [u32; 4],
    transport: [u32; 4],
    controls: [f32; 4],
    flags: [u32; 4],
    acceleration: [u32; 4],
    lights: [u32; 4],
    neural: [u32; 4],
    reconstruction: [f32; 4],
    noise: [f32; 4],
    eye: [f32; 4],
    forward: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
}
/// Linear radiance sums and separate debug sums, identical to the WGSL Pixel layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Pixel {
    pub a: [f32; 4],
    pub b: [f32; 4],
    pub stats_a: [f32; 4],
    pub stats_b: [f32; 4],
    pub normals_a: [f32; 4],
    pub normals_b: [f32; 4],
    pub guide_normal_a: [f32; 4],
    pub guide_albedo_a: [f32; 4],
    pub guide_normal_b: [f32; 4],
    pub guide_albedo_b: [f32; 4],
    pub moment_a: [f32; 4],
    pub moment_b: [f32; 4],
    pub light_a: [f32; 4],
    pub light_b: [f32; 4],
    pub allocation: [f32; 4],
    pub work: [u32; 4],
    pub schedule: [u32; 4],
    pub half_a: [f32; 4],
    pub half_b: [f32; 4],
    pub counts: [u32; 4],
}
pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    uniform: wgpu::Buffer,
    worlds: [GpuWorld; 2],
    pixels: wgpu::Buffer,
    frames: wgpu::Buffer,
    denoiser: Option<denoise::Denoiser>,
    oidn: Option<oidn::Oidn>,
    generation: u64,
    neural: Option<nrc::Neural>,
    caustics: Option<caustics::Caustics>,
    adaptive: Option<adaptive::Adaptive>,
    shader: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    layout: wgpu::BindGroupLayout,
    trace: [wgpu::ComputePipeline; 6],
    present: wgpu::ComputePipeline,
    scene_seed: u32,
    scene_presets: [scene::WorldPreset; 2],
    scene_glass: [scene::GlassPreset; 2],
    previous: Option<Settings>,
    needs_clear: bool,
    display_dirty: bool,
}
impl Renderer {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        width: u32,
        height: u32,
        seed: u32,
    ) -> Result<Self> {
        Self::validate_size(&device, width, height)?;
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transport parameters"),
            size: size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scenes = scene::worlds(seed);
        let worlds = [
            GpuWorld::new(&device, &scenes[0])?,
            GpuWorld::new(&device, &scenes[1])?,
        ];
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Scene query and persistent accumulation"),
            entries: &[
                buffer_entry(0, wgpu::BufferBindingType::Uniform),
                buffer_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                buffer_entry(5, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(6, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(7, wgpu::BufferBindingType::Storage { read_only: false }),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Multiworld integrator"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Stochastic multiworld transport"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}\n{}\n{}",
                    concat!(
                        concat!(
                            include_str!("shaders/trace.wgsl"),
                            "\n",
                            include_str!("shaders/adaptive.wgsl")
                        ),
                        "\n",
                        include_str!("shaders/caustics_stub.wgsl")
                    ),
                    include_str!("shaders/nrc.wgsl"),
                    include_str!("shaders/nrc_scalar.wgsl"),
                    concat!(
                        include_str!("shaders/denoise.wgsl"),
                        "\n",
                        include_str!("shaders/oidn.wgsl")
                    )
                )
                .into(),
            ),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let trace = std::array::from_fn(|mode| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Transport-specialized accumulation"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("accumulate"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("ACTIVE_MODE", mode as f64)],
                    ..Default::default()
                },
                cache: None,
            })
        });
        let present = pipeline("present");
        let (pixels, texture, view) = targets(&device, width, height);
        let frames = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Neural cache and inactive guide input"),
            size: nrc::BYTES + size_of::<denoise::Frame>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = bindings(&device, &layout, &uniform, &worlds, &pixels, &view, &frames);
        Ok(Self {
            device,
            queue,
            width,
            height,
            samples: 0,
            uniform,
            worlds,
            pixels,
            frames,
            denoiser: None,
            oidn: None,
            generation: 0,
            neural: None,
            caustics: None,
            adaptive: None,
            shader,
            pipeline_layout,
            texture,
            view,
            bind_group,
            layout,
            trace,
            present,
            scene_seed: seed,
            scene_presets: [
                scene::WorldPreset::WarmSpheres,
                scene::WorldPreset::CoolBoxes,
            ],
            scene_glass: [scene::GlassPreset::Assorted; 2],
            previous: None,
            needs_clear: true,
            display_dirty: true,
        })
    }
    pub fn validate_size(device: &wgpu::Device, width: u32, height: u32) -> Result<()> {
        let limits = device.limits();
        ensure!(
            width > 0 && height > 0,
            "Render dimensions must be positive"
        );
        ensure!(
            width <= limits.max_texture_dimension_2d && height <= limits.max_texture_dimension_2d,
            "Render dimensions exceed GPU texture limits"
        );
        ensure!(
            nrc::BYTES
                + u64::from(width) * u64::from(height) * (size_of::<Pixel>() as u64).max(128)
                <= limits.max_storage_buffer_binding_size,
            "Resolution exceeds GPU accumulation-buffer limit; try a smaller image"
        );
        ensure!(
            (131_328u64 + 65_536 * 13 + u64::from(width) * u64::from(height) * 6) * 4
                <= limits.max_storage_buffer_binding_size,
            "Resolution exceeds caustic-buffer limit"
        );
        Ok(())
    }
    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        Self::validate_size(&self.device, width, height)?;
        (self.pixels, self.texture, self.view) = targets(&self.device, width, height);
        self.bind_group = bindings(
            &self.device,
            &self.layout,
            &self.uniform,
            &self.worlds,
            &self.pixels,
            &self.view,
            &self.frames,
        );
        self.width = width;
        self.height = height;
        self.denoiser = None;
        self.oidn = None;
        self.neural = None;
        self.caustics = None;
        self.reset();
        Ok(())
    }
    /// Replace both worlds and their acceleration together. Empty and differently
    /// sized worlds are supported. Invalid input leaves the previous pair intact.
    /// Changing the seed later restores the generated worlds and both BVHs.
    pub fn set_worlds(&mut self, scenes: &[Vec<scene::Object>; 2]) -> Result<()> {
        let worlds = [
            GpuWorld::new(&self.device, &scenes[0])?,
            GpuWorld::new(&self.device, &scenes[1])?,
        ];
        self.worlds = worlds;
        self.bind_group = bindings(
            &self.device,
            &self.layout,
            &self.uniform,
            &self.worlds,
            &self.pixels,
            &self.view,
            &self.frames,
        );
        self.reset();
        Ok(())
    }
    pub fn neural_training_batches(&self) -> u32 {
        self.neural.as_ref().map_or(0, |n| n.batches)
    }
    pub fn reset(&mut self) {
        if let Some(neural) = &mut self.neural {
            neural.reset();
        }
        if let Some(denoiser) = &mut self.denoiser {
            denoiser.reset();
        }
        self.clear_accumulation();
    }
    /// Restart the progressive image while retaining a compatible world-space neural model.
    pub fn clear_accumulation(&mut self) {
        self.samples = 0;
        self.generation = self.generation.wrapping_add(1);
        if let Some(oidn) = &mut self.oidn {
            oidn.reset();
        }
        self.needs_clear = true;
        self.display_dirty = true;
    }
    /// Queue order ensures uniforms, accumulation, presentation, and egui see consistent data.
    pub fn render(&mut self, settings: &Settings, advance: bool) {
        self.dispatch(settings, advance, true);
    }
    /// Headless batches defer tone mapping until render(settings, false). No
    /// readback, intermediate display pass, or extra RNG work is needed per batch.
    pub fn accumulate(&mut self, settings: &Settings) {
        self.dispatch(settings, true, false);
    }
    fn dispatch(&mut self, settings: &Settings, advance: bool, display: bool) {
        // Queued inference owns one query per world/pixel. Preserve the public
        // sample budget by ordering one-sample trace/resolve/training steps.
        let adaptive_sampling = settings.sampling != crate::settings::SamplingMode::Uniform;
        let temporal =
            settings.denoise && settings.denoiser == crate::settings::DenoiserKind::Temporal;
        let oidn_display =
            settings.denoise && settings.denoiser == crate::settings::DenoiserKind::Oidn;
        let use_oidn = oidn_display || settings.sampling == crate::settings::SamplingMode::Denoised;
        if let Some(oidn) = &mut self.oidn {
            let ready = oidn.completed_samples();
            oidn.collect(&self.device, &self.queue, self.generation);
            self.display_dirty |= ready != oidn.completed_samples();
        }
        let fast_caustics = settings.caustics_active();
        let neural_cache = settings.neural_cache && !fast_caustics;
        if (neural_cache || fast_caustics || adaptive_sampling) && advance && settings.spp > 1 {
            let single = Settings {
                spp: 1,
                ..settings.clone()
            };
            for i in 0..settings.spp {
                self.dispatch(
                    &single,
                    true,
                    display && (temporal || i + 1 == settings.spp),
                );
            }
            self.previous = Some(settings.clone());
            return;
        }

        if self
            .previous
            .as_ref()
            .is_some_and(|old| settings.invalidates(old))
        {
            if self
                .previous
                .as_ref()
                .is_some_and(|old| settings.history_compatible(old))
            {
                self.clear_accumulation();
            } else {
                self.reset();
            }
        }
        if self.scene_seed != settings.seed
            || self.scene_presets != settings.worlds
            || self.scene_glass != settings.glass
        {
            let scenes =
                scene::selected_worlds_with_glass(settings.seed, settings.worlds, settings.glass);
            self.set_worlds(&scenes)
                .expect("generated geometry must fit the GPU scene limits");
            self.scene_seed = settings.seed;
            self.scene_presets = settings.worlds;
            self.scene_glass = settings.glass;
            self.reset();
        }
        if adaptive_sampling && self.adaptive.is_none() {
            self.adaptive = Some(adaptive::Adaptive::new(
                &self.device,
                &self.shader,
                &self.pipeline_layout,
            ));
        }
        if use_oidn && self.oidn.is_none() {
            self.oidn = Some(oidn::Oidn::new(
                &self.device,
                &self.shader,
                &self.layout,
                [self.width, self.height],
            ));
        }
        if fast_caustics && self.caustics.is_none() {
            self.caustics = Some(caustics::Caustics::new(
                &self.device,
                &self.layout,
                self.width,
                self.height,
            ));
        }
        if neural_cache && self.neural.is_none() {
            self.neural = Some(nrc::Neural::new(
                &self.device,
                &self.shader,
                &self.layout,
                self.width,
                self.height,
            ));
        }
        if temporal && self.denoiser.is_none() {
            let replacement = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Current denoising samples and guides"),
                size: nrc::BYTES
                    + u64::from(self.width)
                        * u64::from(self.height)
                        * size_of::<denoise::Frame>() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut copy = self.device.create_command_encoder(&Default::default());
            copy.copy_buffer_to_buffer(&self.frames, 0, &replacement, 0, nrc::BYTES);
            self.queue.submit([copy.finish()]);
            self.frames = replacement;
            self.bind_group = bindings(
                &self.device,
                &self.layout,
                &self.uniform,
                &self.worlds,
                &self.pixels,
                &self.view,
                &self.frames,
            );
            self.needs_clear = true; // Generate guides even when toggled on while paused.
            self.denoiser = Some(denoise::Denoiser::new(
                &self.device,
                self.width,
                self.height,
                &self.shader,
                &self.layout,
                &self.pipeline_layout,
            ));
        }
        if !temporal && let Some(denoiser) = &mut self.denoiser {
            denoiser.reset();
        }
        let batch = if advance {
            settings.spp.min(16_000_000 - self.samples)
        } else {
            0
        };
        self.display_dirty |= self.previous.as_ref() != Some(settings) || batch > 0;
        let trace = batch > 0 || self.needs_clear;
        let present = display && self.display_dirty;
        let oidn_final = use_oidn
            && display
            && !advance
            && self
                .oidn
                .as_ref()
                .is_some_and(|o| !o.failed && !o.ready(self.samples));
        if !trace && !present && !oidn_final {
            self.previous = Some(settings.clone());
            return;
        }
        let camera = settings.camera(self.width as f32 / self.height as f32);
        let params = Params {
            size: [self.width, self.height, batch, self.samples],
            transport: [
                settings.mode as u32,
                settings.depth,
                settings.seed,
                settings.view as u32,
            ],
            controls: [
                settings.p,
                settings.persistence,
                settings.exposure,
                (if settings.use_bvh {
                    // Avoid mixing traversal algorithms within a SIMD group in
                    // stochastic modes; small-scene fallback is for fixed-world
                    // paths (including the two independent blend estimates).
                    let adaptive = settings.adaptive_bvh
                        && matches!(
                            settings.mode,
                            crate::settings::Mode::WorldA
                                | crate::settings::Mode::WorldB
                                | crate::settings::Mode::Blend
                        );
                    let enabled = |world: &GpuWorld| !adaptive || !world.prefer_linear;
                    (4u32
                        | u32::from(enabled(&self.worlds[0]))
                        | (u32::from(enabled(&self.worlds[1])) << 1)) as f32
                } else {
                    0.0
                }) + if trace && present && !temporal && !neural_cache && !adaptive_sampling {
                    8.0
                } else {
                    0.0
                },
            ],
            flags: [
                settings.spatial as u32,
                settings.primary_a as u32,
                self.worlds[0].object_count,
                self.worlds[1].object_count,
            ],
            acceleration: [
                self.worlds[0].node_count,
                self.worlds[1].node_count,
                self.worlds[0].plane_count,
                self.worlds[1].plane_count,
            ],
            lights: [
                self.worlds[0].light_count,
                self.worlds[1].light_count,
                settings.light_sampling as u32,
                u32::from(self.worlds[0].has_glass) | (u32::from(self.worlds[1].has_glass) << 1),
            ],
            neural: [
                u32::from(neural_cache),
                settings.cache_bounce,
                self.neural.as_ref().map_or(0, |n| n.batches),
                u32::from(settings.cache_self_training),
            ],
            reconstruction: [
                settings.denoise_strength,
                settings.sampling as u32 as f32,
                16.,
                0.,
            ],
            noise: [
                settings.threshold,
                settings.frequency,
                settings.softness,
                settings.time,
            ],
            eye: camera[0],
            forward: camera[1],
            right: camera[2],
            up: camera[3],
        };
        self.queue
            .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&params));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Trace and display"),
            });
        if neural_cache {
            self.neural
                .as_mut()
                .unwrap()
                .initialize(&mut encoder, &self.bind_group);
        }
        if trace && fast_caustics {
            self.caustics.as_ref().unwrap().prepare(
                &self.queue,
                &mut encoder,
                &self.bind_group,
                &self.worlds,
                [self.width, self.height],
                self.samples,
            );
        }
        if let Some(oidn) = &mut self.oidn {
            oidn.publish_feedback(&mut encoder, &self.bind_group);
        }
        if trace && adaptive_sampling {
            self.adaptive.as_ref().unwrap().prepare(
                &mut encoder,
                &self.bind_group,
                &self.pixels,
                [self.width, self.height],
                [self.samples, batch],
                if fast_caustics {
                    self.caustics.as_ref()
                } else {
                    None
                },
            );
        }
        // Clear after reset, even while paused; otherwise skip empty accumulation.
        if trace {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Accumulate linear radiance"),
                timestamp_writes: None,
            });
            if adaptive_sampling {
                if fast_caustics {
                    pass.set_pipeline(
                        self.caustics
                            .as_ref()
                            .unwrap()
                            .adaptive_trace(settings.mode as usize),
                    );
                } else {
                    pass.set_pipeline(self.adaptive.as_mut().unwrap().trace_pipeline(
                        &self.device,
                        &self.shader,
                        &self.pipeline_layout,
                        settings.mode as usize,
                        neural_cache,
                    ));
                }
            } else if fast_caustics {
                pass.set_pipeline(
                    self.caustics
                        .as_ref()
                        .unwrap()
                        .trace_pipeline(settings.mode as usize, temporal),
                );
            } else if neural_cache {
                pass.set_pipeline(
                    self.neural
                        .as_ref()
                        .unwrap()
                        .trace_pipeline(settings.mode as usize, temporal),
                );
            } else if temporal {
                pass.set_pipeline(&self.denoiser.as_ref().unwrap().trace[settings.mode as usize]);
            } else {
                pass.set_pipeline(&self.trace[settings.mode as usize]);
            }
            pass.set_bind_group(0, &self.bind_group, &[]);
            if fast_caustics {
                pass.set_bind_group(1, &self.caustics.as_ref().unwrap().group, &[]);
            } else if neural_cache && !adaptive_sampling {
                pass.set_bind_group(1, &self.neural.as_ref().unwrap().queries, &[]);
            }
            if adaptive_sampling {
                pass.dispatch_workgroups_indirect(&self.adaptive.as_ref().unwrap().indirect, 0);
            } else {
                pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
            }
        }
        if trace && adaptive_sampling && temporal {
            self.adaptive.as_ref().unwrap().guides(
                &mut encoder,
                &self.bind_group,
                [self.width, self.height],
            );
        }
        if trace && neural_cache && !adaptive_sampling {
            self.neural.as_ref().unwrap().resolve(
                &mut encoder,
                &self.bind_group,
                settings.mode as usize,
                self.width,
                self.height,
                temporal,
            );
        }
        if use_oidn
            && (display || settings.sampling == crate::settings::SamplingMode::Denoised)
            && let Some(oidn) = &mut self.oidn
        {
            oidn.request(
                &self.device,
                &mut encoder,
                &self.bind_group,
                oidn::Snapshot {
                    generation: self.generation,
                    samples: self.samples + batch,
                    blend: settings.mode == crate::settings::Mode::Blend,
                    force: display && !advance,
                    uncertainty: settings.sampling == crate::settings::SamplingMode::Denoised,
                    strength: settings.denoise_strength,
                },
            );
        }
        if present && oidn_display && self.oidn.as_ref().is_some_and(|o| o.has_image()) {
            self.oidn
                .as_ref()
                .unwrap()
                .present(&mut encoder, &self.bind_group);
        }
        if present && temporal {
            self.denoiser.as_mut().unwrap().encode(
                &self.queue,
                &mut encoder,
                &self.bind_group,
                settings,
                self.width,
                self.height,
            );
        }
        if present
            && (!trace || neural_cache || adaptive_sampling)
            && !temporal
            && !(oidn_display && self.oidn.as_ref().is_some_and(|o| o.has_image()))
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tone map or debug"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.present);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
        // Tracing sees a frozen model; teachers and training run AFTER it, and
        // publish the next model only for the following dispatch.
        if neural_cache && batch > 0 {
            self.neural.as_mut().unwrap().learn(
                &mut encoder,
                &self.bind_group,
                settings.mode as usize,
                settings.cache_self_training,
            );
        }
        self.queue.submit([encoder.finish()]);
        if let Some(oidn) = &mut self.oidn {
            oidn.after_submit();
        }
        self.samples += batch;
        self.needs_clear = false;
        if present {
            self.display_dirty = false;
        }
        self.previous = Some(settings.clone());
    }
    /// Complete reconstruction for exports. Interactive frames use the worker
    /// asynchronously; this explicit operation waits for the current snapshot.
    pub fn finish_denoising(&mut self, settings: &Settings) -> Result<()> {
        if !settings.denoise
            || settings.denoiser != crate::settings::DenoiserKind::Oidn
            || self.samples == 0
        {
            return Ok(());
        }
        let started = std::time::Instant::now();
        loop {
            self.dispatch(settings, false, true);
            let oidn = self.oidn.as_ref().context("OIDN was not initialized")?;
            ensure!(!oidn.failed, "{}", oidn.status);
            if oidn.ready(self.samples) {
                return Ok(());
            }
            ensure!(
                started.elapsed() < Duration::from_secs(120),
                "Timed out waiting for OIDN"
            );
            self.device.poll(wgpu::PollType::Poll)?;
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    pub fn denoising_status(&self) -> Option<&str> {
        self.oidn.as_ref().map(|o| o.status.as_str())
    }
    fn read_buffer(
        &self,
        encoder: wgpu::CommandEncoder,
        staging: &wgpu::Buffer,
    ) -> Result<Vec<u8>> {
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = mpsc::sync_channel(1);
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(60)),
        })?;
        rx.recv_timeout(Duration::from_secs(60))??;
        let data = staging.slice(..).get_mapped_range()?.to_vec();
        staging.unmap();
        Ok(data)
    }
    /// Explicit raw-buffer readback for exports/tests; allocation decisions stay on GPU.
    pub fn read_pixels(&self) -> Result<Vec<Pixel>> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Linear test readback"),
            size: self.pixels.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.pixels, 0, &staging, 0, self.pixels.size());
        let bytes = self.read_buffer(encoder, &staging)?;
        Ok(bytes
            .as_chunks::<{ size_of::<Pixel>() }>()
            .0
            .iter()
            .map(|chunk| bytemuck::pod_read_unaligned(chunk))
            .collect())
    }
    /// Linear image for numerical comparisons, before exposure/tone mapping.
    /// Raw estimates retain signed control-variate residuals. Reconstructed
    /// estimates match the nonnegative display signal and wait for current OIDN.
    pub fn read_linear_image(
        &mut self,
        settings: &Settings,
        filtered: bool,
    ) -> Result<Vec<[f32; 3]>> {
        if filtered && settings.denoise {
            self.render(settings, false);
            self.finish_denoising(settings)?;
        }
        let count = self.samples.max(1) as f32;
        if filtered
            && settings.denoise
            && settings.denoiser == crate::settings::DenoiserKind::Temporal
        {
            let d = self
                .denoiser
                .as_ref()
                .context("Temporal denoiser was not initialized")?;
            let buffer = d.history[d.current].clone();
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Linear reconstruction readback"),
                size: buffer.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&buffer, 0, &staging, 0, buffer.size());
            let bytes = self.read_buffer(encoder, &staging)?;
            return Ok(bytes
                .as_chunks::<{ size_of::<denoise::Frame>() }>()
                .0
                .iter()
                .map(|p| {
                    let frame: denoise::Frame = bytemuck::pod_read_unaligned(p);
                    std::array::from_fn(|c| frame.sample[c].max(0.))
                })
                .collect());
        }
        let pixels = self.read_pixels()?;
        let raw = |p: &Pixel| {
            std::array::from_fn::<_, 3, _>(|c| {
                if settings.mode == crate::settings::Mode::Blend {
                    (p.b[c] + settings.p * (p.a[c] - p.b[c])) / count
                } else {
                    p.a[c] / count
                }
            })
        };
        if !filtered || !settings.denoise || self.samples == 0 {
            return Ok(pixels.iter().map(raw).collect());
        }
        let output = self
            .oidn
            .as_ref()
            .context("OIDN was not initialized")?
            .output_buffer()
            .clone();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Linear OIDN readback"),
            size: output.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output.size());
        let bytes = self.read_buffer(encoder, &staging)?;
        Ok(bytes
            .as_chunks::<{ size_of::<oidn::Feedback>() }>()
            .0
            .iter()
            .zip(&pixels)
            .map(|(f, p)| {
                let f: oidn::Feedback = bytemuck::pod_read_unaligned(f);
                let original = raw(p);
                std::array::from_fn(|c| {
                    let filtered = if settings.mode == crate::settings::Mode::Blend {
                        f.color_b[c] + settings.p * (f.color_a[c] - f.color_b[c])
                    } else {
                        f.color_a[c]
                    };
                    (original[c] + settings.denoise_strength * (filtered.max(0.) - original[c]))
                        .max(0.)
                })
            })
            .collect())
    }
    pub fn save_png(&mut self, path: &Path) -> Result<()> {
        if let Some(settings) = self.previous.clone() {
            self.finish_denoising(&settings)?;
        }
        self.save_texture_png(&self.texture, path)
    }
    /// Export the unfiltered tone-mapped radiance without changing the live image
    /// or consuming/resetting any temporal history.
    pub fn save_raw_png(&self, path: &Path) -> Result<()> {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Raw PNG export"),
            size: self.texture.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let group = bindings(
            &self.device,
            &self.layout,
            &self.uniform,
            &self.worlds,
            &self.pixels,
            &view,
            &self.frames,
        );
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.present);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
        self.queue.submit([encoder.finish()]);
        self.save_texture_png(&texture, path)
    }
    fn save_texture_png(&self, texture: &wgpu::Texture, path: &Path) -> Result<()> {
        let row = (self.width * 4).div_ceil(256) * 256;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PNG readback"),
            size: u64::from(row) * u64::from(self.height),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(self.height),
                },
            },
            texture.size(),
        );
        let padded = self.read_buffer(encoder, &staging)?;
        let data: Vec<u8> = padded
            .chunks_exact(row as usize)
            .flat_map(|r| r[..(self.width * 4) as usize].iter().copied())
            .collect();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let mut png =
            png::Encoder::new(BufWriter::new(File::create(path)?), self.width, self.height);
        png.set_color(png::ColorType::Rgba);
        png.set_depth(png::BitDepth::Eight);
        png.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
        png.write_header()?.write_image_data(&data)?;
        Ok(())
    }
}
fn buffer_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn targets(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Buffer, wgpu::Texture, wgpu::TextureView) {
    let pixels = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Persistent radiance and debug sums"),
        size: u64::from(width) * u64::from(height) * size_of::<Pixel>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sRGB encoded display"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (pixels, texture, view)
}
fn bindings(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    worlds: &[GpuWorld; 2],
    pixels: &wgpu::Buffer,
    view: &wgpu::TextureView,
    frames: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("World resources"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: worlds[0].geometry.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: worlds[1].geometry.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: pixels.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: worlds[0].nodes.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: worlds[1].nodes.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: frames.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(view),
            },
        ],
    })
}

/// Geometry and acceleration are owned as a pair, so a scene upload cannot leave
/// stale bounds referring to a different world's objects.
struct GpuWorld {
    geometry: wgpu::Buffer,
    nodes: wgpu::Buffer,
    object_count: u32,
    node_count: u32,
    plane_count: u32,
    prefer_linear: bool,
    light_count: u32,
    has_glass: bool,
    glass_targets: Vec<[f32; 4]>,
}
impl GpuWorld {
    fn new(device: &wgpu::Device, objects: &[scene::Object]) -> Result<Self> {
        let mut bvh = Bvh::build(objects)?;
        let node_count = bvh.nodes.len() as u32;
        let glass_targets = caustics::targets(objects, &bvh);
        let (objects, lights) = crate::sampling::prepare(objects);
        let light_count = lights.len() as u32;
        // Compact CDF records share the acceleration allocation after its nodes.
        // Traversal keeps its original logical count; no extra GPU bindings.
        bvh.nodes.extend(lights);
        let objects = objects.as_slice();
        let limit = device
            .limits()
            .max_storage_buffer_binding_size
            .min(device.limits().max_buffer_size);
        ensure!(
            std::mem::size_of_val(objects) as u64 <= limit
                && std::mem::size_of_val(bvh.nodes.as_slice()) as u64 <= limit,
            "World exceeds GPU scene-buffer limits"
        );
        // wgpu storage bindings cannot be zero-sized. Logical counts remain zero.
        let dummy_object = scene::Object::zeroed();
        let dummy_node = crate::bvh::Node::zeroed();
        let geometry = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Per-world geometry"),
            contents: if objects.is_empty() {
                bytemuck::bytes_of(&dummy_object)
            } else {
                bytemuck::cast_slice(objects)
            },
            usage: wgpu::BufferUsages::STORAGE,
        });
        let nodes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Per-world BVH and planes"),
            contents: if bvh.nodes.is_empty() {
                bytemuck::bytes_of(&dummy_node)
            } else {
                bytemuck::cast_slice(&bvh.nodes)
            },
            usage: wgpu::BufferUsages::STORAGE,
        });
        Ok(Self {
            geometry,
            nodes,
            object_count: objects.len() as u32,
            node_count,
            light_count,
            glass_targets,
            has_glass: objects.iter().any(|o| o.color_material[3] >= 2.),
            plane_count: bvh.plane_count,
            // Measured on the target M1: <=32 finite spheres are cheaper to
            // query linearly. Boxes, mixed shapes, and larger worlds use BVHs.
            prefer_linear: objects.len() - bvh.plane_count as usize <= 32
                && objects
                    .iter()
                    .all(|o| o.extent_shape[3] == 0.0 || o.extent_shape[3] == 2.0),
        })
    }
}
#[cfg(test)]
mod tests;
