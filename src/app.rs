use crate::{
    gpu::{self, Renderer},
    scene::WorldPreset,
    settings::{Mode, Settings, View},
};
use anyhow::{Context, Result};
use eframe::egui;
use std::{path::PathBuf, time::Instant};

pub fn run(settings: Settings, width: u32, height: u32, smoke_frames: Option<u32>) -> Result<()> {
    let mut setup = eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.instance_descriptor = gpu::instance_descriptor();
    setup.device_descriptor = std::sync::Arc::new(|_| wgpu::DeviceDescriptor {
        label: Some("Interactive multiworld GPU"),
        ..Default::default()
    });
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
        "Multiworld · stochastic transport lab",
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
                paused: false,
                status: String::new(),
                resolution: [width, height],
                last_frame: Instant::now(),
                fps: 0.,
                frames: 0,
                smoke_frames,
                save_raw_requested: false,
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
    paused: bool,
    status: String,
    resolution: [u32; 2],
    last_frame: Instant,
    fps: f32,
    frames: u32,
    smoke_frames: Option<u32>,
    save_raw_requested: bool,
    capture_dir: String,
}
impl App {
    fn controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut save = false;
        ui.heading("MULTIWORLD");
        ui.label("Stochastic transport lab");
        ui.separator();
        ui.label("Worlds");
        for (slot, label) in [(0, "World A"), (1, "World B")] {
            ui.horizontal(|ui| {
                ui.label(label);
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
        }
        if ui.button("Swap A ↔ B").clicked() {
            self.settings.worlds.swap(0, 1);
        }
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
        ui.separator();
        ui.checkbox(
            &mut self.settings.light_sampling,
            "Importance sampling (MIS)",
        )
        .on_hover_text(
            "Power-weighted lights, solid-angle sampling, and stratified world selection.",
        );
        ui.checkbox(&mut self.settings.use_bvh, "BVH acceleration");
        if self.settings.use_bvh {
            ui.checkbox(
                &mut self.settings.adaptive_bvh,
                "Automatic small-scene fallback",
            );
        }
        ui.checkbox(
            &mut self.settings.denoise,
            "Temporal denoise (display only)",
        );
        if self.settings.denoise {
            ui.add(
                egui::Slider::new(&mut self.settings.denoise_strength, 0.0..=1.0)
                    .text("Denoise strength"),
            );
            ui.small("Dual-world history; raw samples stay available for comparison.");
        }
        ui.add(egui::Slider::new(&mut self.settings.spp, 1..=16).text("Samples / frame"));
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
            save = ui.button("Save PNG").clicked();
        });
        self.save_raw_requested = ui.button("Save raw PNG").clicked();
        ui.collapsing("Export folder", |ui| {
            ui.text_edit_singleline(&mut self.capture_dir);
        });
        egui::ComboBox::from_label("View")
            .selected_text(self.settings.view.label())
            .show_ui(ui, |ui| {
                for view in View::ALL {
                    ui.selectable_value(&mut self.settings.view, view, view.label());
                }
            });
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
            "{:.1} FPS · {} spp\n{} × {}",
            self.fps, self.renderer.samples, self.renderer.width, self.renderer.height
        ));
        let info = self.state.adapter.get_info();
        ui.small(format!("{} / {:?}", info.name, info.backend));
        ui.small("Drag image to orbit · scroll to zoom");
        ui.small("Hybrid transport is an artificial process.");
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
        self.renderer.render(&self.settings, !self.paused);
        if self.save_raw_requested {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            let path = PathBuf::from(&self.capture_dir)
                .join(format!("{}-raw-{stamp}.png", self.settings.mode.slug()));
            self.status = match self.renderer.save_raw_png(&path) {
                Ok(()) => format!("Saved {}", path.display()),
                Err(e) => format!("Raw export failed: {e:#}"),
            };
            self.save_raw_requested = false;
        }
        if save {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            let path = PathBuf::from(&self.capture_dir).join(format!(
                "{}-{}spp-{stamp}.png",
                self.settings.mode.slug(),
                self.renderer.samples
            ));
            self.status = match self.renderer.save_png(&path) {
                Ok(()) => format!("Saved {}", path.display()),
                Err(e) => format!("Export failed: {e:#}"),
            };
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
        if !self.paused {
            ui.ctx().request_repaint();
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
