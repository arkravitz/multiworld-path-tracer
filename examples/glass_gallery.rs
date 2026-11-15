//! Actual path-traced online-mesh glass captures, including focused caustics.
use anyhow::Result;
use clap::Parser;
use multiworld_path_tracer::{
    gpu::{self, Renderer},
    scene::{GlassPreset, WorldPreset},
    settings::{Mode, Settings},
};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 640)]
    width: u32,
    #[arg(long, default_value_t = 480)]
    height: u32,
    #[arg(long, default_value_t = 1024)]
    samples: u32,
    #[arg(long, default_value = "output/glass-gallery")]
    output: PathBuf,
    #[arg(long)]
    single: bool,
    #[arg(long)]
    denoise: bool,
    #[arg(long)]
    caustics: bool,
    #[arg(long)]
    only: Option<String>,
}
fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        args.samples > 0 && args.samples <= 16384,
        "Invalid sample budget"
    );
    let (device, queue) = pollster::block_on(gpu::headless_device())?;
    let mut r = Renderer::new(device, queue, args.width, args.height, 42)?;
    std::fs::create_dir_all(&args.output)?;
    let mut report = String::new();
    for (i, (name, world, glass)) in [
        (
            "01-faceted-teapot",
            WorldPreset::GlassGallery,
            GlassPreset::Assorted,
        ),
        (
            "02-emerald-teapot",
            WorldPreset::GlassGallery,
            GlassPreset::Emerald,
        ),
        (
            "03-crystal-suzanne",
            WorldPreset::CrystalGarden,
            GlassPreset::Assorted,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        if args.only.as_ref().is_some_and(|selected| selected != name) {
            continue;
        }
        if args.single && i > 0 {
            break;
        }
        let mut s = Settings {
            worlds: [world, WorldPreset::CrystalGarden],
            glass: [glass, GlassPreset::Assorted],
            mode: Mode::WorldA,
            depth: 20,
            spp: 8,
            yaw: 0.6,
            pitch: 0.55,
            distance: 13.,
            exposure: 0.25,
            denoise: args.denoise,
            caustics: args.caustics,
            ..Default::default()
        };
        r.reset();
        let started = Instant::now();
        while r.samples < args.samples {
            s.spp = (args.samples - r.samples).min(8);
            r.accumulate(&s);
            if r.samples.is_multiple_of(64) {
                r.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(60)),
                })?;
            }
        }
        r.render(&s, false);
        r.save_png(&args.output.join(format!("{name}.png")))?;
        let row = format!(
            "{name}: {}x{}, {} spp, {:.2}s; {:?}, {:?}; denoise={}; caustics={}\n",
            args.width,
            args.height,
            args.samples,
            started.elapsed().as_secs_f64(),
            world,
            glass,
            args.denoise,
            args.caustics
        );
        eprint!("{row}");
        report.push_str(&row);
    }
    std::fs::write(args.output.join("render-info.txt"), report)?;
    Ok(())
}
