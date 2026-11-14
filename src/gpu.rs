mod denoise;
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
    Ok(adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Multiworld compute device"),
            ..Default::default()
        })
        .await?)
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
                    "{}\n{}",
                    include_str!("shaders/trace.wgsl"),
                    include_str!("shaders/denoise.wgsl")
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
            label: Some("Inactive denoiser input"),
            size: size_of::<denoise::Frame>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
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
            u64::from(width) * u64::from(height) * size_of::<denoise::Frame>() as u64
                <= limits.max_storage_buffer_binding_size,
            "Resolution exceeds GPU accumulation-buffer limit; try 1280x720 or lower"
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
    pub fn reset(&mut self) {
        if let Some(denoiser) = &mut self.denoiser {
            denoiser.reset();
        }
        self.clear_accumulation();
    }
    fn clear_accumulation(&mut self) {
        self.samples = 0;
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
        if self.scene_seed != settings.seed || self.scene_presets != settings.worlds {
            let scenes = scene::selected_worlds(settings.seed, settings.worlds);
            self.set_worlds(&scenes)
                .expect("generated geometry must fit the GPU scene limits");
            self.scene_seed = settings.seed;
            self.scene_presets = settings.worlds;
            self.reset();
        }
        if settings.denoise && self.denoiser.is_none() {
            self.frames = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Current denoising samples and guides"),
                size: u64::from(self.width)
                    * u64::from(self.height)
                    * size_of::<denoise::Frame>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
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
        if !settings.denoise
            && let Some(denoiser) = &mut self.denoiser
        {
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
        if !trace && !present {
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
                }) + if trace && present && !settings.denoise {
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
                0,
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
        // Clear after reset, even while paused; otherwise skip empty accumulation.
        if trace {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Accumulate linear radiance"),
                timestamp_writes: None,
            });
            if settings.denoise {
                pass.set_pipeline(&self.denoiser.as_ref().unwrap().trace[settings.mode as usize]);
            } else {
                pass.set_pipeline(&self.trace[settings.mode as usize]);
            }
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
        if present && settings.denoise {
            self.denoiser.as_mut().unwrap().encode(
                &self.queue,
                &mut encoder,
                &self.bind_group,
                settings,
                self.width,
                self.height,
            );
        }
        if present && !trace && !settings.denoise {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tone map or debug"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.present);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
        self.queue.submit([encoder.finish()]);
        self.samples += batch;
        self.needs_clear = false;
        if present {
            self.display_dirty = false;
        }
        self.previous = Some(settings.clone());
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
    /// Only explicit exports/tests read data back; ordinary interactive frames stay on GPU.
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
    pub fn save_png(&self, path: &Path) -> Result<()> {
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
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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
}
impl GpuWorld {
    fn new(device: &wgpu::Device, objects: &[scene::Object]) -> Result<Self> {
        let mut bvh = Bvh::build(objects)?;
        let node_count = bvh.nodes.len() as u32;
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
