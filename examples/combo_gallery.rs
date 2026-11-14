//! Art-directed pairings captured by the actual renderer. No image compositing.
use anyhow::{Result, ensure};
use clap::Parser;
use multiworld_path_tracer::{
    gpu::{self, Renderer},
    scene::WorldPreset::{self, *},
    settings::{Mode, Settings},
};
use std::{
    collections::HashSet,
    path::PathBuf,
    time::{Duration, Instant},
};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 960)]
    width: u32,
    #[arg(long, default_value_t = 720)]
    height: u32,
    #[arg(long, default_value_t = 1024)]
    samples: u32,
    #[arg(long, default_value = "output/combo-gallery")]
    output: PathBuf,
    /// Render the full contact-sheet exploration instead of the selected gallery.
    #[arg(long)]
    explore: bool,
    /// Capture specific shots, separated by commas.
    #[arg(long, value_delimiter = ',')]
    only: Vec<String>,
}
fn spatial(pair: [WorldPreset; 2], yaw: f32, time: f32, frequency: f32) -> Settings {
    Settings {
        worlds: pair,
        mode: Mode::Persistent,
        spatial: true,
        persistence: 0.88,
        yaw,
        pitch: 0.40,
        distance: 16.,
        time,
        frequency,
        softness: 0.14,
        exposure: 0.75,
        spp: 8,
        depth: 12,
        ..Default::default()
    }
}
fn borrowed(pair: [WorldPreset; 2], yaw: f32) -> Settings {
    Settings {
        worlds: pair,
        mode: Mode::PerBounce,
        primary_a: true,
        p: 0.16,
        yaw,
        pitch: 0.40,
        distance: 16.,
        exposure: 0.95,
        spp: 8,
        depth: 12,
        ..Default::default()
    }
}
fn shots() -> Vec<(&'static str, Settings)> {
    vec![
        (
            "01-overgrown-sanctuary",
            spatial([SunkenTemple, BioluminescentGrove], 0.7, 3., 0.62),
        ),
        (
            "02-neon-orrery",
            spatial([LunarObservatory, NeonDistrict], -0.7, 2.4, 0.9),
        ),
        (
            "03-electric-ruins",
            borrowed([SunkenTemple, NeonDistrict], 0.62),
        ),
        (
            "04-alien-planetarium",
            spatial([LunarObservatory, BioluminescentGrove], 2.2, 4., 0.75),
        ),
        (
            "05-midnight-forest",
            Settings {
                exposure: 1.2,
                ..borrowed([BioluminescentGrove, NeonDistrict], -0.85)
            },
        ),
        (
            "06-reclaimed-district",
            borrowed([NeonDistrict, BioluminescentGrove], 0.7),
        ),
        (
            "07-cosmic-forest",
            borrowed([BioluminescentGrove, LunarObservatory], 0.7),
        ),
        (
            "08-fractured-temple",
            spatial([SunkenTemple, LunarObservatory], 2.6, -2., 1.2),
        ),
        ("09-amber-city", borrowed([NeonDistrict, SunkenTemple], 1.2)),
        (
            "10-chrome-cathedral",
            Settings {
                pitch: 0.39,
                distance: 15.5,
                ..borrowed([LunarObservatory, SunkenTemple], 0.62)
            },
        ),
        (
            "11-portal-garden",
            spatial([NeonDistrict, BioluminescentGrove], 0.62, 5.8, 0.55),
        ),
        (
            "12-overgrown-sanctuary-orbit",
            Settings {
                pitch: 0.5,
                persistence: 0.94,
                exposure: 0.9,
                ..spatial([SunkenTemple, BioluminescentGrove], -1.1, 1.5, 0.8)
            },
        ),
        (
            "13-neon-orrery-orbit",
            Settings {
                persistence: 0.94,
                softness: 0.12,
                exposure: 0.8,
                ..spatial([LunarObservatory, NeonDistrict], 2.3, -3., 0.7)
            },
        ),
        (
            "14-living-stone",
            borrowed([SunkenTemple, BioluminescentGrove], -1.),
        ),
        (
            "15-two-realities",
            Settings {
                spatial: false,
                mode: Mode::PerBounce,
                ..spatial([LunarObservatory, NeonDistrict], 0.7, 0., 0.35)
            },
        ),
        (
            "16-neon-in-chrome",
            Settings {
                exposure: 1.3,
                ..borrowed([WarmSpheres, NeonDistrict], 0.62)
            },
        ),
        (
            "17-forest-portal-close",
            Settings {
                distance: 14.5,
                pitch: 0.39,
                exposure: 1.0,
                ..spatial([SunkenTemple, BioluminescentGrove], 0.2, 3.8, 0.5)
            },
        ),
        (
            "18-orbital-ruins",
            spatial([LunarObservatory, SunkenTemple], 0.7, 7., 0.65),
        ),
    ]
}
const SELECTED: &[&str] = &[
    "03-electric-ruins",
    "05-midnight-forest",
    "18-orbital-ruins",
    "11-portal-garden",
    "15-two-realities",
    "17-forest-portal-close",
];
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (1..=65536).contains(&args.samples),
        "samples must be in 1..=65536"
    );
    let all = shots();
    let names: HashSet<_> = all.iter().map(|(name, _)| *name).collect();
    ensure!(
        names.len() == all.len(),
        "Gallery shot names must be unique"
    );
    for name in &args.only {
        ensure!(all.iter().any(|(n, _)| n == name), "Unknown shot {name}");
    }
    let (device, queue) = pollster::block_on(gpu::headless_device())?;
    let mut renderer = Renderer::new(device, queue, args.width, args.height, 42)?;
    std::fs::create_dir_all(&args.output)?;
    let mut manifest = format!(
        "# Combination gallery settings\n\nActual GPU renders; MIS on, denoising off, seed 42, depth 12. Exposure is applied before Reinhard tone mapping and sRGB output. No compositing or retouching.\n\nGPU: {:?}\n\n",
        renderer.device.adapter_info()
    );
    for (name, mut s) in all {
        let selected = if !args.only.is_empty() {
            args.only.iter().any(|n| n == name)
        } else {
            args.explore || SELECTED.contains(&name)
        };
        if !selected {
            continue;
        }
        renderer.reset();
        let start = Instant::now();
        while renderer.samples < args.samples {
            s.spp = 8.min(args.samples - renderer.samples);
            renderer.accumulate(&s);
            if renderer.samples.is_multiple_of(64) {
                renderer.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(60)),
                })?;
            }
        }
        renderer.render(&s, false);
        renderer.save_png(&args.output.join(format!("{name}.png")))?;
        manifest.push_str(&format!(
            "## {name}\n\n{}×{}, {} spp.\n\n```text\n{s:#?}\n```\n\n",
            args.width, args.height, args.samples
        ));
        std::fs::write(args.output.join("settings.md"), &manifest)?;
        eprintln!("Saved {name}: {:.2}s", start.elapsed().as_secs_f64());
    }
    Ok(())
}
