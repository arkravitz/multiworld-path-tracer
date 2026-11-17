use anyhow::{Result, ensure};
use clap::Parser;
use multiworld_path_tracer::{
    app,
    gpu::{self, Renderer},
    scene::{GlassPreset, WorldPreset},
    settings::{DenoiserKind, Mode, SamplingMode, Settings},
};
use std::{path::PathBuf, time::Instant};
#[derive(Parser, Debug)]
#[command(version, about = "GPU stochastic path tracing across multiple worlds")]
struct Args {
    #[arg(long)]
    headless: bool,
    /// Use the original cosine-only estimator for comparison.
    #[arg(long)]
    no_light_sampling: bool,
    /// Use the linear intersection reference instead of the per-world BVHs.
    #[arg(long)]
    brute_force: bool,
    /// Use software traversal for comparison even when hardware ray queries are available.
    #[arg(long)]
    software_tracing: bool,
    /// Force BVH traversal even for small sphere-only worlds.
    #[arg(long, conflicts_with = "brute_force")]
    force_bvh: bool,
    /// Filter display/PNG only; raw linear estimates remain unchanged.
    #[arg(long)]
    denoise: bool,
    #[arg(long, value_enum, default_value = "oidn")]
    denoiser: DenoiserKind,
    #[arg(long, value_enum, default_value = "uniform")]
    sampling: SamplingMode,
    /// Experimental biased online neural radiance cache; fused GPU kernels.
    #[arg(long)]
    neural_cache: bool,
    /// Compare neural caches using the previous SIMD/scalar inference path.
    #[arg(long)]
    simd_neural: bool,
    /// Targeted light tracing and progressive photon gathering (fixed worlds/blend).
    #[arg(long)]
    caustics: bool,
    /// Photon-driven neural radiance caching (SPPC), fixed worlds/blend only.
    #[arg(long)]
    photon_nrc: bool,
    #[arg(long,default_value_t=32768,value_parser=clap::value_parser!(u32).range(256..=524288))]
    photon_nrc_photons: u32,
    #[arg(long,default_value_t=2048,value_parser=clap::value_parser!(u32).range(1024..=32768))]
    photon_nrc_queries: u32,
    #[arg(long,default_value_t=14,value_parser=clap::value_parser!(u32).range(10..=19))]
    photon_nrc_hash_log2: u32,
    #[arg(long, default_value_t = 0.25)]
    photon_nrc_global_radius: f32,
    #[arg(long, default_value_t = 0.08)]
    photon_nrc_caustic_radius: f32,
    /// Learn light/caster selection from earlier photon contributions.
    #[arg(long, requires = "caustics", conflicts_with = "no_photon_guiding")]
    photon_guiding: bool,
    /// Restore uniform caster selection instead of the default learned guide.
    #[arg(long, requires = "caustics")]
    no_photon_guiding: bool,
    /// Weighted temporal and neighboring-reservoir reuse of light-path seeds.
    #[arg(long, requires = "caustics")]
    photon_reuse: bool,
    /// Newton-solved sphere transmission paths guide normalized emission cones.
    #[arg(long, requires = "caustics")]
    manifold_guiding: bool,
    /// Light paths per active world and dispatch; zero selects automatic sizing.
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..=32768))]
    photon_budget: u32,
    #[arg(long, default_value_t = 0)]
    cache_bounce: u32,
    /// Shorten training paths using learned lighting; retain 1/16 uncached anchors.
    #[arg(long, requires = "neural_cache")]
    cache_self_training: bool,
    /// Render all six modes using the same camera, seed, and sample budget.
    #[arg(long, requires = "headless")]
    all_modes: bool,
    #[arg(long, value_enum, default_value = "per-bounce")]
    mode: Mode,
    #[arg(long, value_enum, default_value = "sunken-temple")]
    world_a: WorldPreset,
    #[arg(long, value_enum, default_value = "bioluminescent-grove")]
    world_b: WorldPreset,
    #[arg(long, value_enum, default_value = "assorted")]
    glass_a: GlassPreset,
    #[arg(long, value_enum, default_value = "assorted")]
    glass_b: GlassPreset,
    #[arg(long, default_value_t = 800)]
    width: u32,
    #[arg(long, default_value_t = 600)]
    height: u32,
    /// Average camera samples per pixel in headless mode (per world for blend).
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
    ensure!(
        (0..=63).contains(&args.cache_bounce),
        "--cache-bounce must be in 0..=63 (0 uses path spread)"
    );
    ensure!(
        args.sampling != SamplingMode::Denoised || args.denoiser == DenoiserKind::Oidn,
        "Denoising-aware sampling requires the OIDN denoiser"
    );
    ensure!(args.exposure.is_finite(), "Exposure must be finite");
    let mut settings = Settings {
        mode: args.mode,
        worlds: [args.world_a, args.world_b],
        glass: [args.glass_a, args.glass_b],
        light_sampling: !args.no_light_sampling,
        use_bvh: !args.brute_force,
        hardware_ray_tracing: !args.software_tracing && !args.force_bvh,
        adaptive_bvh: !args.force_bvh,
        denoise: args.denoise || args.sampling == SamplingMode::Denoised,
        denoiser: args.denoiser,
        sampling: args.sampling,
        neural_cache: args.neural_cache,
        tensor_neural: !args.simd_neural,
        caustics: args.caustics,
        photon_nrc: args.photon_nrc,
        photon_nrc_photons: args.photon_nrc_photons,
        photon_nrc_queries: args.photon_nrc_queries,
        photon_nrc_hash_log2: args.photon_nrc_hash_log2,
        photon_nrc_global_radius: args.photon_nrc_global_radius,
        photon_nrc_caustic_radius: args.photon_nrc_caustic_radius,
        photon_guiding: args.photon_guiding || (args.caustics && !args.no_photon_guiding),
        photon_reuse: args.photon_reuse,
        manifold_guiding: args.manifold_guiding,
        photon_budget: args.photon_budget,
        cache_bounce: args.cache_bounce,
        cache_self_training: args.cache_self_training,
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
    ensure!(
        !args.photon_nrc
            || (matches!(args.mode, Mode::WorldA | Mode::WorldB | Mode::Blend)
                && args.sampling == SamplingMode::Uniform
                && !args.all_modes),
        "Photon NRC supports fixed worlds/blend and uniform sampling"
    );
    ensure!(
        args.photon_nrc_global_radius.is_finite()
            && args.photon_nrc_global_radius >= 0.01
            && args.photon_nrc_global_radius <= 2.
            && args.photon_nrc_caustic_radius.is_finite()
            && args.photon_nrc_caustic_radius >= 0.005
            && args.photon_nrc_caustic_radius <= 2.,
        "Invalid photon NRC radii"
    );
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
        "Multiworld GPU comparison\nBVH={}, adaptive={}, denoise={}, light_sampling={}\n{}x{}, {} spp (per world for blend), seed {}, depth {}\np={}, s={}, spatial={}, primary_a={}\nCamera: yaw={}, pitch={}, distance={}\nExposure={} EV\nGPU: {:?}\n",
        settings.use_bvh,
        settings.adaptive_bvh,
        settings.denoise,
        settings.light_sampling,
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
    report.push_str(&format!(
        "World A: {}\nWorld B: {}\n",
        settings.worlds[0].label(),
        settings.worlds[1].label()
    ));
    report.push_str(&format!(
        "Glass A: {}; Glass B: {}\n",
        settings.glass[0].label(),
        settings.glass[1].label()
    ));
    report.push_str(&format!(
        "Sampling: {:?}; denoiser: {:?}\n",
        settings.sampling, settings.denoiser
    ));
    report.push_str(&format!(
        "Photon guiding: {}; reservoir reuse: {}; manifold-guided spheres: {}; photon budget: {} (0=auto)\n",
        settings.photon_guiding, settings.photon_reuse, settings.manifold_guiding, settings.photon_budget
    ));
    report.push_str(&format!(
        "Neural cache: {}, cache bounce: {}, self-training: {}\n",
        settings.neural_cache, settings.cache_bounce, settings.cache_self_training
    ));
    report.push_str(&format!(
        "Photon NRC: {}; photons {}; queries {}; hash log2 {}; radii {}/{}\n",
        settings.photon_nrc,
        settings.photon_nrc_photons,
        settings.photon_nrc_queries,
        settings.photon_nrc_hash_log2,
        settings.photon_nrc_global_radius,
        settings.photon_nrc_caustic_radius
    ));
    for mode in modes {
        settings.mode = mode;
        report.push_str(&format!(
            "Fast caustics requested: {}; active: {}\n",
            settings.caustics,
            settings.caustics_active()
        ));
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
