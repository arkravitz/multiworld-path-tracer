//! Equal-wall-clock, raw-linear comparison against an independent camera reference.
use anyhow::Result;
use clap::Parser;
use multiworld_path_tracer::{
    gpu::{self, Pixel, Renderer},
    scene::WorldPreset,
    settings::{Mode, Settings},
};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 128)]
    width: u32,
    #[arg(long, default_value_t = 96)]
    height: u32,
    #[arg(long, default_value_t = 8192)]
    reference_samples: u32,
    #[arg(long, default_value_t = 2.0)]
    seconds: f64,
    #[arg(long, value_enum, default_value = "glass-gallery")]
    world: WorldPreset,
    #[arg(long, default_value = "output/caustics-benchmark")]
    output: PathBuf,
}
fn wait(r: &Renderer) -> Result<()> {
    r.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(60)),
    })?;
    Ok(())
}
fn image(r: &Renderer) -> Result<Vec<[f64; 3]>> {
    Ok(r.read_pixels()?
        .into_iter()
        .map(|p: Pixel| std::array::from_fn(|i| p.a[i] as f64 / r.samples as f64))
        .collect())
}
fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        args.seconds.is_finite()
            && args.seconds > 0.
            && (1..=16_000_000).contains(&args.reference_samples),
        "Positive time and reference sample count required"
    );
    let (device, queue) = pollster::block_on(gpu::headless_device())?;
    let mut r = Renderer::new(device, queue, args.width, args.height, 42)?;
    let mut s = Settings {
        mode: Mode::WorldA,
        worlds: [args.world, args.world],
        depth: 20,
        yaw: 0.6,
        pitch: 0.55,
        distance: 13.,
        exposure: 0.25,
        ..Default::default()
    };
    std::fs::create_dir_all(&args.output)?;
    // An independent seed avoids correlated reference noise. Then restore the
    // test seed and warm both estimators, including first-use Metal compilation.
    s.seed = 1347;
    // The scene seed also rotates the mesh. Hold geometry at seed 42 while
    // changing only the reference ray streams, otherwise MSE compares scenes.
    r.render(&s, false);
    let worlds = multiworld_path_tracer::scene::selected_worlds_with_glass(42, s.worlds, s.glass);
    r.set_worlds(&worlds)?;
    while r.samples < args.reference_samples {
        s.spp = (args.reference_samples - r.samples).min(64);
        r.accumulate(&s);
        wait(&r)?;
    }
    let reference = image(&r)?;
    let floor: Vec<_> = r
        .read_pixels()?
        .iter()
        .map(|p| p.normals_a[1] / r.samples as f32 > 0.995)
        .collect();
    r.render(&s, false);
    r.save_png(&args.output.join("reference.png"))?;
    s.seed = 42;
    for enabled in [false, true] {
        s.caustics = enabled;
        s.spp = 1;
        r.render(&s, true);
        wait(&r)?;
    }
    let mut report =
        String::from("estimator,samples,seconds,linear_nrmse,floor_nrmse,linear_mean\n");
    for enabled in [false, true] {
        s.caustics = enabled;
        s.spp = 1;
        r.reset();
        let start = Instant::now();
        loop {
            s.spp = if enabled { 1 } else { 8 };
            r.accumulate(&s);
            wait(&r)?;
            if start.elapsed().as_secs_f64() >= args.seconds {
                break;
            }
        }
        let seconds = start.elapsed().as_secs_f64();
        let pixels = image(&r)?;
        let error: f64 = pixels
            .iter()
            .zip(&reference)
            .flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).powi(2)))
            .sum();
        let energy: f64 = reference.iter().flatten().map(|v| v * v).sum();
        let floor_error: f64 = pixels
            .iter()
            .zip(&reference)
            .zip(&floor)
            .filter(|(_, mask)| **mask)
            .flat_map(|((a, b), _)| (0..3).map(move |c| (a[c] - b[c]).powi(2)))
            .sum();
        let floor_energy: f64 = reference
            .iter()
            .zip(&floor)
            .filter(|(_, mask)| **mask)
            .flat_map(|(p, _)| p.iter().map(|v| v * v))
            .sum();
        let mean: f64 = pixels.iter().flatten().sum::<f64>() / (pixels.len() * 3) as f64;
        let name = if enabled {
            "light-tracing"
        } else {
            "camera-paths"
        };
        let row = format!(
            "{name},{},{seconds:.6},{:.6},{:.6},{mean:.6}\n",
            r.samples,
            (error / energy).sqrt(),
            (floor_error / floor_energy).sqrt()
        );
        eprint!("{row}");
        report.push_str(&row);
        r.render(&s, false);
        r.save_png(&args.output.join(format!("{name}.png")))?;
    }
    std::fs::write(args.output.join("timing.csv"), report)?;
    std::fs::write(
        args.output.join("configuration.txt"),
        format!(
            "{}x{}; {:?}; depth 20; raw linear; no cache/denoiser; {} spp camera reference ray seed 1347; geometry seed 42; test ray seed 42; warm pipelines excluded, all GPU work waited; target {}s. NRMSE uses a finite noisy reference, not ground truth.\n",
            args.width, args.height, args.world, args.reference_samples, args.seconds
        ),
    )?;
    Ok(())
}
