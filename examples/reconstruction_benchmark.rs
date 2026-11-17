//! Equal-wall-time comparison of reconstruction and pixel allocation. OIDN is
//! warmed outside timing; completing the CURRENT denoised image stays inside it.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use multiworld_path_tracer::{
    gpu::{self, Renderer},
    scene::{self, WorldPreset},
    settings::{DenoiserKind, Mode, SamplingMode, Settings, View},
};
use std::{
    hash::{Hash, Hasher},
    path::PathBuf,
    time::{Duration, Instant},
};
#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 160)]
    width: u32,
    #[arg(long, default_value_t = 120)]
    height: u32,
    #[arg(long, default_value_t = 4096)]
    reference_samples: u32,
    #[arg(long, default_value_t = 2.0)]
    seconds: f64,
    #[arg(long, value_enum, default_value = "glass-gallery")]
    world: WorldPreset,
    #[arg(long)]
    no_caustics: bool,
    #[arg(long, default_value_t = 0.8)]
    denoise_strength: f32,
    #[arg(long)]
    reuse_reference: bool,
    #[arg(long, default_value = "output/reconstruction-benchmark")]
    output: PathBuf,
}
fn wait(r: &Renderer) -> Result<()> {
    r.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(60)),
    })?;
    Ok(())
}
fn error(a: &[[f32; 3]], b: &[[f32; 3]]) -> f64 {
    let mut difference = 0_f64;
    let mut energy = 0_f64;
    for (a, b) in a.iter().zip(b) {
        for c in 0..3 {
            difference += (a[c] as f64 - b[c] as f64).powi(2);
            energy += (b[c] as f64).powi(2);
        }
    }
    (difference / energy.max(1e-30)).sqrt()
}
fn display_error(a: &[[f32; 3]], b: &[[f32; 3]], exposure: f32) -> f64 {
    let map = |v: f32| {
        let l = (v * exposure.exp2()).max(0.);
        let v = l / (1. + l);
        if v <= 0.0031308 {
            12.92 * v
        } else {
            1.055 * v.powf(1. / 2.4) - 0.055
        }
    };
    let sum: f64 = a
        .iter()
        .zip(b)
        .flat_map(|(a, b)| (0..3).map(move |c| (map(a[c]) - map(b[c])).powi(2) as f64))
        .sum();
    (sum / (a.len() * 3) as f64).sqrt()
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (0.0..=1.0).contains(&args.denoise_strength),
        "Denoise strength must be in [0,1]"
    );
    ensure!(
        args.seconds.is_finite()
            && args.seconds > 0.
            && (1..=16_000_000).contains(&args.reference_samples),
        "Invalid time or reference sample budget"
    );
    let (device, queue) = pollster::block_on(gpu::headless_device())?;
    let mut r = Renderer::new(device, queue, args.width, args.height, 42)?;
    let mut settings = Settings {
        mode: Mode::WorldA,
        worlds: [args.world, args.world],
        depth: 20,
        yaw: 0.6,
        pitch: 0.55,
        distance: 13.,
        exposure: 0.25,
        denoise_strength: args.denoise_strength,
        ..Default::default()
    };
    let worlds = scene::selected_worlds_with_glass(42, settings.worlds, settings.glass);
    let mut hasher = std::hash::DefaultHasher::new();
    for world in &worlds {
        bytemuck::cast_slice::<_, u8>(world).hash(&mut hasher);
    }
    let reference_key = format!(
        "reconstruction-v1; {}x{}; {} spp; geometry {:x}; ray seed 1347; depth 20; yaw .6 pitch .55 distance 13; camera-only reference\n",
        args.width,
        args.height,
        args.reference_samples,
        hasher.finish()
    );
    std::fs::create_dir_all(&args.output)?;
    let reference_file = args.output.join("reference-linear.bin");
    let reference_config = args.output.join("reference-key.txt");
    let reference: Vec<[f32; 3]> = if args.reuse_reference {
        ensure!(
            std::fs::read_to_string(&reference_config)? == reference_key,
            "Cached reference does not match the scene/camera/budget"
        );
        let bytes = std::fs::read(&reference_file)?;
        ensure!(
            bytes.len() == args.width as usize * args.height as usize * 12,
            "Invalid cached reference size"
        );
        bytes
            .as_chunks::<12>()
            .0
            .iter()
            .map(|p| bytemuck::pod_read_unaligned(p))
            .collect()
    } else {
        settings.seed = 1347;
        r.render(&settings, false);
        r.set_worlds(&worlds)?;
        while r.samples < args.reference_samples {
            settings.spp = (args.reference_samples - r.samples).min(32);
            r.accumulate(&settings);
            wait(&r)?;
        }
        let image = r.read_linear_image(&settings, false)?;
        r.render(&settings, false);
        r.save_png(&args.output.join("reference.png"))?;
        std::fs::write(&reference_file, bytemuck::cast_slice(&image))?;
        std::fs::write(&reference_config, &reference_key)?;
        image
    };
    settings.seed = 42;
    settings.caustics = !args.no_caustics;
    let mut report = String::from(
        "method,camera_spp,seconds,raw_nrmse,reconstructed_nrmse,display_rmse,min_pixel_paths,max_pixel_paths\n",
    );
    for (name, denoise, kind, sampling) in [
        (
            "raw-uniform",
            false,
            DenoiserKind::Oidn,
            SamplingMode::Uniform,
        ),
        (
            "temporal-uniform",
            true,
            DenoiserKind::Temporal,
            SamplingMode::Uniform,
        ),
        (
            "oidn-uniform",
            true,
            DenoiserKind::Oidn,
            SamplingMode::Uniform,
        ),
        (
            "oidn-variance",
            true,
            DenoiserKind::Oidn,
            SamplingMode::Variance,
        ),
        (
            "oidn-aware",
            true,
            DenoiserKind::Oidn,
            SamplingMode::Denoised,
        ),
    ] {
        settings.denoise = denoise;
        settings.denoiser = kind;
        settings.sampling = sampling;
        settings.view = View::Radiance;
        settings.spp = 4;
        r.reset();
        for _ in 0..4 {
            r.render(&settings, true);
            wait(&r)?;
        }
        r.finish_denoising(&settings)?;
        wait(&r)?;
        r.reset();
        let start = Instant::now();
        loop {
            r.render(&settings, true);
            wait(&r)?;
            if start.elapsed().as_secs_f64() >= args.seconds {
                break;
            }
        }
        r.render(&settings, false);
        r.finish_denoising(&settings)?;
        wait(&r)?;
        let elapsed = start.elapsed().as_secs_f64();
        let raw = r.read_linear_image(&settings, false)?;
        let reconstruction = r.read_linear_image(&settings, true)?;
        let counts = r.read_pixels()?;
        let total: u64 = counts.iter().map(|p| u64::from(p.counts[0])).sum();
        ensure!(
            total == u64::from(args.width) * u64::from(args.height) * u64::from(r.samples),
            "Camera ray budget was not preserved: actual {total}, expected {}, spp {}, min {}, max {}, guards {}",
            u64::from(args.width) * u64::from(args.height) * u64::from(r.samples),
            r.samples,
            counts.iter().map(|p| p.counts[0]).min().unwrap_or(0),
            counts.iter().map(|p| p.counts[0]).max().unwrap_or(0),
            counts.iter().map(|p| u64::from(p.counts[2])).sum::<u64>()
        );
        ensure!(
            counts.iter().all(|p| p.counts[2] == 0),
            "Adaptive work corruption guard fired"
        );
        let min = counts
            .iter()
            .map(|p| p.counts[0])
            .min()
            .context("empty image")?;
        let max = counts.iter().map(|p| p.counts[0]).max().unwrap();
        let line = format!(
            "{name},{},{elapsed:.6},{:.6},{:.6},{:.6},{min},{max}\n",
            r.samples,
            error(&raw, &reference),
            error(&reconstruction, &reference),
            display_error(&reconstruction, &reference, settings.exposure)
        );
        eprint!("{line}");
        report.push_str(&line);
        r.save_png(&args.output.join(format!("{name}.png")))?;
        r.save_raw_png(&args.output.join(format!("{name}-raw.png")))?;
        if sampling != SamplingMode::Uniform {
            settings.view = View::Samples;
            r.render(&settings, false);
            r.save_png(&args.output.join(format!("{name}-allocation.png")))?;
        }
        std::fs::write(args.output.join("timing.csv"), &report)?;
    }
    std::fs::write(
        args.output.join("configuration.txt"),
        format!(
            "{reference_key}Test seed 42; world {:?}; caustics {}; denoiser strength {}; warm pipelines/models excluded; GPU completion, interactive denoising and final current-snapshot reconstruction included; NRMSE uses finite independent camera reference, not ground truth. Frame budgets may overrun the target by one batch and final reconstruction. Each camera_spp is an average; allocation heatmaps show actual paths.\n",
            args.world, !args.no_caustics, args.denoise_strength
        ),
    )?;
    Ok(())
}
