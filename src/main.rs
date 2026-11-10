use anyhow::{Result, ensure};
use clap::Parser;
use multiworld_path_tracer::{
    app,
    gpu::{self, Renderer},
    settings::{Mode, Settings},
};
use std::{path::PathBuf, time::Instant};
#[derive(Parser, Debug)]
#[command(version, about = "GPU stochastic path tracing across multiple worlds")]
struct Args {
    #[arg(long)]
    headless: bool,
    /// Use the linear intersection reference instead of the per-world BVHs.
    #[arg(long)]
    brute_force: bool,
    /// Force BVH traversal even for small sphere-only worlds.
    #[arg(long, conflicts_with = "brute_force")]
    force_bvh: bool,
    /// Filter display/PNG only; raw linear estimates remain unchanged.
    #[arg(long)]
    denoise: bool,
    /// Render all six modes using the same camera, seed, and sample budget.
    #[arg(long, requires = "headless")]
    all_modes: bool,
    #[arg(long, value_enum, default_value = "per-bounce")]
    mode: Mode,
    #[arg(long, default_value_t = 800)]
    width: u32,
    #[arg(long, default_value_t = 600)]
    height: u32,
    /// Total samples per pixel in headless mode (blend traces this many in EACH world).
    #[arg(long, default_value_t = 256)]
    samples: u32,
    #[arg(long, default_value_t = 1)]
    spp: u32,
    #[arg(long, default_value_t = 10)]
    bounces: u32,
    #[arg(long, default_value_t = 0.5)]
    probability: f32,
    #[arg(long, default_value_t = 0.8)]
    persistence: f32,
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    exposure: f32,
    #[arg(long, default_value_t = 42)]
    seed: u32,
    #[arg(long, default_value = "output")]
    output: PathBuf,
    #[arg(long)]
    spatial: bool,
    #[arg(long)]
    primary_a: bool,
    #[arg(long, hide = true)]
    smoke_frames: Option<u32>,
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (1..=16_000_000).contains(&args.samples),
        "--samples must be in 1..=16000000"
    );
    ensure!((1..=64).contains(&args.spp), "--spp must be in 1..=64");
    ensure!(
        (1..=64).contains(&args.bounces),
        "--bounces must be in 1..=64"
    );
    ensure!(
        (0.0..=1.0).contains(&args.probability) && (0.0..=1.0).contains(&args.persistence),
        "Probability and persistence must be finite and in [0,1]"
    );
    ensure!(args.exposure.is_finite(), "Exposure must be finite");
    let mut settings = Settings {
        mode: args.mode,
        use_bvh: !args.brute_force,
        adaptive_bvh: !args.force_bvh,
        denoise: args.denoise,
        p: args.probability,
        persistence: args.persistence,
        spp: args.spp,
        depth: args.bounces,
        exposure: args.exposure,
        seed: args.seed,
        spatial: args.spatial,
        primary_a: args.primary_a,
        ..Default::default()
    };
    if !args.headless {
        return app::run(settings, args.width, args.height, args.smoke_frames);
    }
    let (device, queue) = pollster::block_on(gpu::headless_device())?;
    let mut renderer = Renderer::new(device, queue, args.width, args.height, args.seed)?;
    let modes = if args.all_modes {
        Mode::ALL.to_vec()
    } else {
        vec![args.mode]
    };
    std::fs::create_dir_all(&args.output)?;
    let mut report = format!(
        "Multiworld GPU comparison\nBVH={}, adaptive={}, denoise={}\n{}x{}, {} spp (per world for blend), seed {}, depth {}\np={}, s={}, spatial={}, primary_a={}\nCamera: yaw={}, pitch={}, distance={}\nExposure={} EV\nGPU: {:?}\n",
        settings.use_bvh,
        settings.adaptive_bvh,
        settings.denoise,
        args.width,
        args.height,
        args.samples,
        args.seed,
        args.bounces,
        settings.p,
        settings.persistence,
        settings.spatial,
        settings.primary_a,
        settings.yaw,
        settings.pitch,
        settings.distance,
        settings.exposure,
        renderer.device.adapter_info()
    );
    for mode in modes {
        settings.mode = mode;
        renderer.reset();
        let start = Instant::now();
        while renderer.samples < args.samples {
            settings.spp = args.spp.min(args.samples - renderer.samples);
            renderer.accumulate(&settings);
            // Bound outstanding work in long headless jobs without per-frame image readback.
            if renderer.samples.is_multiple_of(64 * args.spp) {
                renderer.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(60)),
                })?;
            }
        }
        renderer.render(&settings, false);
        let path = args
            .output
            .join(format!("{}-{}spp.png", mode.slug(), args.samples));
        renderer.save_png(&path)?;
        let line = format!(
            "{}: {:.2}s -> {}\n",
            mode.label(),
            start.elapsed().as_secs_f32(),
            path.display()
        );
        eprint!("{line}");
        report.push_str(&line);
    }
    std::fs::write(args.output.join("render-info.txt"), report)?;
    Ok(())
}
