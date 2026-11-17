use crate::{
    gpu::{self, Renderer},
    scene::{GlassPreset, WorldPreset},
    settings::{DenoiserKind, Mode, SamplingMode, Settings},
};
use anyhow::{Context, Result};
use eframe::egui;
use std::{path::PathBuf, time::Instant};
mod labs;
use labs::{Lab, Labs, glass_settings};

pub fn run(
    mut settings: Settings,
    width: u32,
    height: u32,
    smoke_frames: Option<u32>,
) -> Result<()> {
    let labs = Labs::new(&mut settings);
    let mut setup = eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.instance_descriptor = gpu::instance_descriptor();
    setup.device_descriptor = std::sync::Arc::new(gpu::device_descriptor);
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200., 820.])
            .with_min_inner_size([850., 600.]),
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(setup),
            ..Default::default()
        },
        ..Default::default()
    };
    eframe::run_native(
        "Multiworld · transport labs",
        options,
        Box::new(move |cc| {
            let state = cc
                .wgpu_render_state
                .as_ref()
                .context("eframe did not initialize wgpu")?
                .clone();
            let renderer = Renderer::new(
                state.device.clone(),
                state.queue.clone(),
                width,
                height,
                settings.seed,
            )?;
            let texture = state.renderer.write().register_native_texture(
                &state.device,
                &renderer.view,
                wgpu::FilterMode::Linear,
            );
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(App {
                renderer,
                state,
                texture,
                settings,
                labs,
                paused: false,
                status: String::new(),
                resolution: [width, height],
                last_frame: Instant::now(),
                fps: 0.,
                frames: 0,
                smoke_frames,
                save_raw_requested: false,
                pending_export: None,
                writing_export: None,
                capture_dir: default_capture_directory().to_string_lossy().into_owned(),
            }))
        }),
    )
    .map_err(|e| anyhow::anyhow!("Window initialization failed: {e}"))
}
struct App {
    renderer: Renderer,
    state: eframe::egui_wgpu::RenderState,
    texture: egui::TextureId,
    settings: Settings,
    labs: Labs,
    paused: bool,
    status: String,
    resolution: [u32; 2],
    last_frame: Instant,
    fps: f32,
    frames: u32,
    smoke_frames: Option<u32>,
    save_raw_requested: bool,
    pending_export: Option<PendingExport>,
    writing_export: Option<(PathBuf, gpu::export::ImageExport)>,
    capture_dir: String,
}
struct PendingExport {
    path: PathBuf,
    raw: bool,
    settings: Settings,
    generation: Option<u64>,
    started: Instant,
}
impl App {
    fn controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut save = false;
        ui.heading("TRANSPORT LABS");
        let mut selected = self.labs.active;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut selected, Lab::Multiworld, "Multiworld");
            ui.selectable_value(&mut selected, Lab::Glass, "Glass & caustics");
        });
        if self.labs.switch(selected, &mut self.settings) {
            self.renderer.reset();
            self.status.clear();
        }
        let multiworld = self.labs.active == Lab::Multiworld;
        ui.label(if multiworld {
            "Multiworld lab · light across incompatible worlds"
        } else {
            "Glass & caustics lab · refraction and focused light"
        });
        if !multiworld {
            egui::ComboBox::from_label("Composition")
                .selected_text(match self.settings.mode {
                    Mode::WorldB => "Scene B",
                    Mode::Blend => "Blend two independent scenes",
                    _ => "Scene A",
                })
                .show_ui(ui, |ui| {
                    for (mode, label) in [
                        (Mode::WorldA, "Scene A"),
                        (Mode::WorldB, "Scene B"),
                        (Mode::Blend, "Blend two independent scenes"),
                    ] {
                        ui.selectable_value(&mut self.settings.mode, mode, label);
                    }
                });
            if self.settings.mode == Mode::Blend {
                ui.add(egui::Slider::new(&mut self.settings.p, 0.0..=1.0).text("Scene A weight"));
            }
            if ui.button("Reset glass showcase").clicked() {
                self.settings = glass_settings(self.settings.seed);
                self.renderer.reset();
            }
        }
        ui.separator();
        ui.label(if multiworld {
            "Worlds"
        } else {
            "Scenes & glass"
        });
        for (slot, label) in [(0, "World A"), (1, "World B")] {
            if !multiworld
                && ((self.settings.mode == Mode::WorldA && slot == 1)
                    || (self.settings.mode == Mode::WorldB && slot == 0))
            {
                continue;
            }
            ui.horizontal(|ui| {
                ui.label(if multiworld {
                    label
                } else if slot == 0 {
                    "Scene A"
                } else {
                    "Scene B"
                });
                egui::ComboBox::from_id_salt(("world-preset", slot))
                    .selected_text(self.settings.worlds[slot].label())
                    .width(210.)
                    .show_ui(ui, |ui| {
                        for preset in WorldPreset::ALL {
                            ui.selectable_value(
                                &mut self.settings.worlds[slot],
                                preset,
                                preset.label(),
                            )
                            .on_hover_text(preset.description());
                        }
                    });
            });
            ui.small(self.settings.worlds[slot].description());
            if self.settings.worlds[slot].has_glass() {
                egui::ComboBox::from_id_salt(("glass-preset", slot))
                    .selected_text(self.settings.glass[slot].label())
                    .show_ui(ui, |ui| {
                        for glass in GlassPreset::ALL {
                            ui.selectable_value(
                                &mut self.settings.glass[slot],
                                glass,
                                glass.label(),
                            );
                        }
                    });
            }
        }
        if (multiworld || self.settings.mode == Mode::Blend) && ui.button("Swap A ↔ B").clicked()
        {
            self.settings.worlds.swap(0, 1);
            self.settings.glass.swap(0, 1);
        }
        if multiworld {
            ui.collapsing("Suggested pairings", |ui| {
                for (label, pair) in [
                    (
                        "Temple × grove",
                        [WorldPreset::SunkenTemple, WorldPreset::BioluminescentGrove],
                    ),
                    (
                        "Observatory × neon",
                        [WorldPreset::LunarObservatory, WorldPreset::NeonDistrict],
                    ),
                    (
                        "Temple × neon",
                        [WorldPreset::SunkenTemple, WorldPreset::NeonDistrict],
                    ),
                    (
                        "Original spheres × boxes",
                        [WorldPreset::WarmSpheres, WorldPreset::CoolBoxes],
                    ),
                ] {
                    if ui.button(label).clicked() {
                        self.settings.worlds = pair;
                    }
                }
            });
            ui.separator();
            egui::ComboBox::from_label("Transport")
                .selected_text(self.settings.mode.label())
                .show_ui(ui, |ui| {
                    for mode in Mode::ALL {
                        ui.selectable_value(&mut self.settings.mode, mode, mode.label());
                    }
                });
            ui.add(egui::Slider::new(&mut self.settings.p, 0.0..=1.0).text("P(World A)"));
            ui.add_enabled(
                self.settings.mode == Mode::Persistent,
                egui::Slider::new(&mut self.settings.persistence, 0.0..=1.0).text("Persistence s"),
            );
            if self.settings.mode == Mode::Blend {
                ui.small("Blend weight updates without restarting A or B.");
            }
        }
        ui.separator();
        ui.checkbox(
            &mut self.settings.light_sampling,
            "Importance sampling (MIS)",
        )
        .on_hover_text(
            "Power-weighted lights, solid-angle sampling, and stratified world selection.",
        );
        if !multiworld {
            if ui
                .checkbox(
                    &mut self.settings.photon_nrc,
                    "Photon-driven neural cache (paper)",
                )
                .changed()
            {
                self.settings.sampling = SamplingMode::Uniform;
                self.settings.neural_cache = false;
                self.settings.caustics = !self.settings.photon_nrc;
            }
            if self.settings.photon_nrc {
                ui.small("SPPC photon teachers · hash-encoded cache · experimental");
                ui.small(format!(
                    "{} training rounds",
                    self.renderer.photon_nrc_batches()
                ));
                ui.add(
                    egui::DragValue::new(&mut self.settings.photon_nrc_photons)
                        .range(256..=524288)
                        .speed(1024)
                        .prefix("Photons "),
                );
                ui.add(
                    egui::DragValue::new(&mut self.settings.photon_nrc_queries)
                        .range(1024..=32768)
                        .speed(1024)
                        .prefix("Queries "),
                );
                ui.add(
                    egui::Slider::new(&mut self.settings.photon_nrc_global_radius, 0.01..=1.)
                        .text("Global radius"),
                );
                ui.add(
                    egui::Slider::new(&mut self.settings.photon_nrc_caustic_radius, 0.005..=0.5)
                        .text("Caustic radius"),
                );
                ui.small("Photon-trained lighting with path corrections to preserve shadows and fine caustics. Uncached preview after camera motion.");
            }
            ui.add_enabled_ui(!self.settings.photon_nrc, |ui| {
                ui.checkbox(&mut self.settings.caustics, "Fast glass caustics (experimental)")
                .on_hover_text("Aim light at glass. Sharp visible caustics; progressive photon gathering through glass.");
                if self.settings.caustics {
                    ui.small("Photon gathering improves over time. Compare raw export to check reconstruction.");
                    ui.checkbox(
                        &mut self.settings.photon_guiding,
                        "Learn useful photon launches",
                    );
                    ui.checkbox(
                        &mut self.settings.photon_reuse,
                        "Reuse successful light paths",
                    );
                    ui.checkbox(
                        &mut self.settings.manifold_guiding,
                        "Manifold-guided spheres",
                    );
                    if self.settings.manifold_guiding {
                        ui.small("Analytic sphere transmission solver; meshes keep ordinary photon proposals.");
                    }
                    ui.horizontal(|ui| {
                        ui.label("Photon budget");
                        ui.add(
                            egui::DragValue::new(&mut self.settings.photon_budget)
                                .range(0..=32768)
                                .speed(256),
                        );
                    });
                    ui.small("0 = automatic. Budget is per world and rendering dispatch.");
                }
            });
        }
        if multiworld {
            ui.checkbox(
                &mut self.settings.neural_cache,
                "Neural radiance cache (experimental)",
            )
            .on_hover_text(
                "Online GPU-trained MLP. Biased indirect lighting; compare with cache disabled.",
            );
            if self.settings.neural_cache {
                ui.checkbox(
                &mut self.settings.cache_self_training,
                "Shorter training paths (experimental)",
            )
            .on_hover_text(
                "Use learned lighting to shorten training rays. Keeps 1/16 fully traced; may trade accuracy for lower training cost.",
            );
                let mut automatic = self.settings.cache_bounce == 0;
                if ui
                    .checkbox(&mut automatic, "Automatic path-spread termination")
                    .changed()
                {
                    self.settings.cache_bounce = if automatic { 0 } else { 2 };
                }
                if !automatic {
                    ui.add(
                        egui::Slider::new(&mut self.settings.cache_bounce, 1..=6)
                            .text("First cache bounce"),
                    );
                }
                ui.small(format!(
                    "Fused GPU training · {} batches",
                    self.renderer.neural_training_batches()
                ));
                ui.small("Approximate indirect light. Reset clears the learned cache.");
            }
        }
        if (self.settings.neural_cache || self.settings.photon_nrc)
            && self.renderer.tensor_inference_available()
        {
            ui.checkbox(&mut self.settings.tensor_neural, "M5 tensor inference")
                .on_hover_text(
                    "Batched neural inference using Metal TensorOps and half-precision inputs.",
                );
            ui.small(self.renderer.neural_inference_backend(&self.settings));
        }
        ui.checkbox(&mut self.settings.use_bvh, "BVH acceleration");
        if self.settings.use_bvh {
            ui.add_enabled(
                self.renderer.hardware_ray_tracing_available(),
                egui::Checkbox::new(
                    &mut self.settings.hardware_ray_tracing,
                    "Hardware ray tracing",
                ),
            );
            ui.checkbox(
                &mut self.settings.adaptive_bvh,
                "Automatic small-scene fallback",
            );
        }
        ui.checkbox(&mut self.settings.denoise, "Denoise");
        if self.settings.denoise {
            let previous_denoiser = self.settings.denoiser;
            egui::ComboBox::from_label("Denoiser")
                .selected_text(self.settings.denoiser.label())
                .show_ui(ui, |ui| {
                    for kind in DenoiserKind::ALL {
                        ui.selectable_value(&mut self.settings.denoiser, kind, kind.label());
                    }
                });
            if previous_denoiser != self.settings.denoiser
                && self.settings.denoiser == DenoiserKind::Temporal
                && self.settings.sampling == SamplingMode::Denoised
            {
                self.settings.sampling = SamplingMode::Variance;
            }
            if self.settings.denoiser == DenoiserKind::Oidn
                && let Some(status) = self.renderer.denoising_status()
            {
                ui.small(status);
            }
            ui.add(
                egui::Slider::new(&mut self.settings.denoise_strength, 0.0..=1.0)
                    .text("Denoise strength"),
            );
            ui.small("Raw samples stay available for comparison.");
        }
        let old_sampling = self.settings.sampling;
        egui::ComboBox::from_label("Sampling")
            .selected_text(self.settings.sampling.label())
            .show_ui(ui, |ui| {
                for mode in SamplingMode::ALL {
                    ui.add_enabled_ui(
                        !self.settings.photon_nrc_active() || mode == SamplingMode::Uniform,
                        |ui| {
                            ui.selectable_value(&mut self.settings.sampling, mode, mode.label());
                        },
                    );
                }
            });
        if self.settings.sampling != old_sampling
            && self.settings.sampling == SamplingMode::Denoised
        {
            self.settings.denoise = true;
            self.settings.denoiser = DenoiserKind::Oidn;
        }
        if self.settings.sampling != SamplingMode::Uniform {
            ui.small("More paths for uncertain pixels; a fixed total ray budget.");
            if self.settings.sampling == SamplingMode::Denoised && !self.settings.denoise {
                ui.small("OIDN still guides adaptive sampling.");
            }
        }
        let single_job = self.settings.photon_nrc_active()
            || self.settings.caustics_active()
            || self.settings.neural_cache
            || self.settings.sampling != SamplingMode::Uniform;
        ui.add_enabled(
            !single_job,
            egui::Slider::new(&mut self.settings.spp, 1..=16).text(
                if self.settings.sampling == SamplingMode::Uniform {
                    "Samples / frame"
                } else {
                    "Average samples / frame"
                },
            ),
        );
        if single_job {
            ui.small("One sample per interactive job keeps controls responsive.");
        }
        ui.add(egui::Slider::new(&mut self.settings.depth, 1..=32).text("Max bounces"));
        ui.add(egui::Slider::new(&mut self.settings.exposure, -5.0..=5.0).text("Exposure (EV)"));
        ui.horizontal(|ui| {
            ui.label("Seed");
            ui.add(egui::DragValue::new(&mut self.settings.seed));
        });
        ui.horizontal(|ui| {
            if ui
                .button(if self.paused {
                    "▶ Resume"
                } else {
                    "Ⅱ Pause"
                })
                .clicked()
            {
                self.paused = !self.paused;
            }
            if ui.button("Reset").clicked() {
                self.renderer.reset();
            }
            save = ui
                .add_enabled(
                    self.pending_export.is_none() && self.writing_export.is_none(),
                    egui::Button::new("Save PNG"),
                )
                .clicked();
        });
        self.save_raw_requested = ui
            .add_enabled(
                self.pending_export.is_none() && self.writing_export.is_none(),
                egui::Button::new("Save raw PNG"),
            )
            .clicked();
        ui.collapsing("Export folder", |ui| {
            ui.text_edit_singleline(&mut self.capture_dir);
        });
        egui::ComboBox::from_label("View")
            .selected_text(self.settings.view.label())
            .show_ui(ui, |ui| {
                for &view in self.labs.views() {
                    ui.selectable_value(&mut self.settings.view, view, view.label());
                }
            });
        if multiworld {
            ui.collapsing("World-space probability", |ui| {
            ui.checkbox(&mut self.settings.spatial, "Use 3D value noise");
            ui.add(egui::Slider::new(&mut self.settings.threshold, 0.0..=1.0).text("Threshold"));
            ui.add(egui::Slider::new(&mut self.settings.frequency, 0.01..=2.0).text("Frequency"));
            ui.add(egui::Slider::new(&mut self.settings.softness, 0.01..=0.5).text("Softness"));
            ui.add(egui::Slider::new(&mut self.settings.time, -10.0..=10.0).text("Time offset"));
            ui.small("Primary: first shared surface. Later: ray origin. Replaces constant p; no automatic animation.");
        });
            ui.checkbox(&mut self.settings.primary_a, "Primary rays always in A");
            ui.small("Applies to mixed path modes (4–6).");
        }
        ui.collapsing("Camera & resolution", |ui| {
            ui.horizontal(|ui| {
                ui.label("Orbit");
                let mut degrees = self.settings.yaw.to_degrees();
                if ui
                    .add(egui::DragValue::new(&mut degrees).speed(0.5).suffix("°"))
                    .changed()
                {
                    self.settings.set_orbit_yaw(degrees.to_radians());
                }
            });
            ui.small("Drag Orbit or the image continuously; horizontal rotation wraps.");
            ui.add(egui::Slider::new(&mut self.settings.pitch, 0.05..=1.4).text("Elevation"));
            ui.add(egui::Slider::new(&mut self.settings.distance, 5.0..=40.0).text("Distance"));
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut self.resolution[0]).range(64..=4096));
                ui.label("×");
                ui.add(egui::DragValue::new(&mut self.resolution[1]).range(64..=4096));
            });
            if ui.button("Apply resolution").clicked() {
                match self.renderer.resize(self.resolution[0], self.resolution[1]) {
                    Ok(()) => {
                        self.state
                            .renderer
                            .write()
                            .update_egui_texture_from_wgpu_texture(
                                &self.state.device,
                                &self.renderer.view,
                                wgpu::FilterMode::Linear,
                                self.texture,
                            );
                        self.status.clear();
                    }
                    Err(e) => self.status = e.to_string(),
                }
            }
        });
        ui.separator();
        ui.monospace(format!(
            "UI {:.1} FPS · {} {}\n{} × {}",
            self.fps,
            self.renderer.samples,
            if self.settings.sampling == SamplingMode::Uniform {
                "spp"
            } else {
                "avg spp"
            },
            self.renderer.width,
            self.renderer.height
        ));
        let info = self.state.adapter.get_info();
        ui.small(format!("{} / {:?}", info.name, info.backend));
        ui.small("Drag image to orbit · scroll to zoom");
        if multiworld {
            ui.small("Hybrid transport is an artificial process.");
        }
        ui.small(concat!("Version ", env!("CARGO_PKG_VERSION")));
        if !self.status.is_empty() {
            ui.label(&self.status);
        }
        save
    }
}
impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.fps = 0.9 * self.fps + 0.1 / dt.max(0.0001);
        let mut save = false;
        egui::Panel::left("controls")
            .resizable(false)
            .show(ui, |ui| {
                ui.set_width(290.);
                egui::ScrollArea::vertical().show(ui, |ui| {
                    save = self.controls(ui);
                });
            });
        egui::CentralPanel::default().show(ui, |ui| {
            let available = ui.available_size();
            let aspect = self.renderer.width as f32 / self.renderer.height as f32;
            let size = egui::vec2(
                available.x.min(available.y * aspect),
                available.y.min(available.x / aspect),
            );
            let response =
                ui.add(egui::Image::new((self.texture, size)).sense(egui::Sense::drag()));
            if response.dragged() {
                let d = ui.input(|i| i.pointer.delta());
                self.settings.orbit(-d.x * 0.007, d.y * 0.007);
            }
            if response.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                self.settings.distance =
                    (self.settings.distance * (-scroll * 0.002).exp()).clamp(5., 40.);
            }
        });
        if save || self.save_raw_requested {
            let raw = self.save_raw_requested;
            self.save_raw_requested = false;
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            let suffix = if raw { "raw" } else { "display" };
            let path = PathBuf::from(&self.capture_dir).join(format!(
                "{}-{suffix}-{stamp}.png",
                self.settings.mode.slug()
            ));
            self.pending_export = Some(PendingExport {
                path,
                raw,
                settings: self.settings.clone(),
                generation: None,
                started: Instant::now(),
            });
            self.status = "Preparing image export…".into();
        }
        if self.pending_export.as_ref().is_some_and(|p| {
            p.settings != self.settings
                || p.generation
                    .is_some_and(|g| g != self.renderer.image_generation())
        }) {
            self.pending_export = None;
            self.status = "Export canceled because render settings changed.".into();
        }
        let export_needs_sample = self.pending_export.as_ref().is_some_and(|p| {
            p.generation.is_none() && !self.renderer.has_accumulation_for(&p.settings)
        });
        let applied = self.renderer.render_live(
            &self.settings,
            export_needs_sample || (!self.paused && self.pending_export.is_none()),
        );
        if let Some(pending) = &mut self.pending_export {
            if applied && pending.generation.is_none() {
                pending.generation = Some(self.renderer.image_generation());
            }
            if pending.started.elapsed() > std::time::Duration::from_secs(120) {
                self.pending_export = None;
                self.status = "Image export timed out.".into();
            } else if pending.generation.is_some()
                && self
                    .renderer
                    .image_export_ready(&self.settings, pending.raw)
            {
                let pending = self.pending_export.take().unwrap();
                match self.renderer.start_image_export(&pending.path, pending.raw) {
                    Ok(job) => {
                        self.writing_export = Some((pending.path, job));
                        self.status = "Writing PNG…".into();
                    }
                    Err(error) => self.status = format!("Export failed: {error:#}"),
                }
            }
        }
        if let Some((path, job)) = &self.writing_export
            && let Some(result) = job.poll()
        {
            self.status = match result {
                Ok(()) => format!("Saved {}", path.display()),
                Err(error) => format!("Export failed: {error:#}"),
            };
            self.writing_export = None;
        }
        self.frames += 1;
        if self.smoke_frames.is_some_and(|n| self.frames >= n) {
            if let Err(e) = self
                .renderer
                .save_png(std::path::Path::new("output/interactive-smoke.png"))
            {
                eprintln!("Smoke export: {e:#}");
            }
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }
        if !self.paused
            || self.renderer.has_live_work()
            || self.pending_export.is_some()
            || self.writing_export.is_some()
        {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(8));
        }
    }
}

fn default_capture_directory() -> PathBuf {
    if let Some(path) = std::env::var_os("MULTIWORLD_CAPTURE_DIR") {
        return PathBuf::from(path);
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    // Finder launches app bundles with `/` as CWD, which is read-only on macOS.
    // Use a user-owned Pictures folder instead of trying to write /captures.
    if cwd.parent().is_none() {
        if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
            return PathBuf::from(home)
                .join("Pictures")
                .join("Multiworld Path Tracer");
        }
        return std::env::temp_dir().join("multiworld-captures");
    }
    cwd.join("captures")
}
