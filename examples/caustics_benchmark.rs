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
    /// Compare the three optional methods separately and together.
    #[arg(long)]
    advanced: bool,
    #[arg(long)]
    photon_nrc: bool,
    #[arg(long)]
    simd_neural: bool,
    #[arg(long, default_value_t = 32768)]
    photon_nrc_photons: u32,
    #[arg(long, default_value_t = 2048)]
    photon_nrc_queries: u32,
    #[arg(long, default_value_t = 14)]
    photon_nrc_hash_log2: u32,
    /// Excluded cache learning time; zero measures cold startup too.
    #[arg(long, default_value_t = 0.)]
    cache_warmup_seconds: f64,
    #[arg(long, default_value_t = 0)]
    photon_budget: u32,
    /// Import a validated camera reference from reconstruction_benchmark.
    #[arg(long)]
    reference_dir: Option<PathBuf>,
    #[arg(long, value_delimiter = ',')]
    only: Vec<String>,
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
        tensor_neural: !args.simd_neural,
        worlds: [args.world, args.world],
        photon_budget: args.photon_budget,
        photon_nrc_photons: args.photon_nrc_photons,
        photon_nrc_queries: args.photon_nrc_queries,
        photon_nrc_hash_log2: args.photon_nrc_hash_log2,
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
    let reference = if let Some(dir) = &args.reference_dir {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::hash::DefaultHasher::new();
        for world in &worlds {
            bytemuck::cast_slice::<_, u8>(world).hash(&mut hasher);
        }
        let expected = format!(
            "reconstruction-v1; {}x{}; {} spp; geometry {:x}; ray seed 1347; depth 20; yaw .6 pitch .55 distance 13; camera-only reference\n",
            args.width,
            args.height,
            args.reference_samples,
            hasher.finish()
        );
        anyhow::ensure!(
            std::fs::read_to_string(dir.join("reference-key.txt"))? == expected,
            "Reference geometry/camera/budget mismatch"
        );
        let bytes = std::fs::read(dir.join("reference-linear.bin"))?;
        anyhow::ensure!(
            bytes.len() == args.width as usize * args.height as usize * 12,
            "Invalid reference length"
        );
        s.spp = 1;
        r.accumulate(&s);
        wait(&r)?;
        bytes
            .as_chunks::<12>()
            .0
            .iter()
            .map(|chunk| {
                let value: [f32; 3] = bytemuck::pod_read_unaligned(chunk);
                value.map(f64::from)
            })
            .collect::<Vec<_>>()
    } else {
        while r.samples < args.reference_samples {
            s.spp = (args.reference_samples - r.samples).min(64);
            r.accumulate(&s);
            wait(&r)?;
        }
        image(&r)?
    };
    let floor: Vec<_> = r
        .read_pixels()?
        .iter()
        .map(|p| p.normals_a[1] / r.samples as f32 > 0.995)
        .collect();
    r.render(&s, false);
    if let Some(dir) = &args.reference_dir {
        std::fs::copy(dir.join("reference.png"), args.output.join("reference.png"))?;
    } else {
        r.save_png(&args.output.join("reference.png"))?;
    }
    s.seed = 42;
    let mut methods = vec![
        ("camera-paths", false, false, false, false),
        ("light-tracing", true, false, false, false),
    ];
    if args.advanced {
        methods.extend([
            ("photon-guiding", true, true, false, false),
            ("reservoir-reuse", true, false, true, false),
            ("manifold-guiding", true, false, false, true),
            ("guided-reuse", true, true, true, false),
            ("guided-manifold", true, true, false, true),
            ("reuse-manifold", true, false, true, true),
            ("combined", true, true, true, true),
        ]);
    }
    if args.photon_nrc {
        methods.push(("photon-nrc", false, false, false, false));
    }
    for requested in &args.only {
        anyhow::ensure!(
            methods.iter().any(|m| m.0 == requested),
            "Unknown method {requested}"
        );
    }
    if !args.only.is_empty() {
        methods.retain(|m| args.only.iter().any(|name| name == m.0));
    }
    for &(name, enabled, guided, reuse, manifold) in &methods {
        s.photon_nrc = name == "photon-nrc";
        s.caustics = enabled;
        s.photon_guiding = guided;
        s.photon_reuse = reuse;
        s.manifold_guiding = manifold;
        s.spp = 1;
        r.render(&s, true);
        wait(&r)?;
    }
    let mut report =
        String::from("estimator,samples,seconds,linear_nrmse,floor_nrmse,linear_mean\n");
    for &(name, enabled, guided, reuse, manifold) in &methods {
        s.photon_nrc = name == "photon-nrc";
        s.caustics = enabled;
        s.photon_guiding = guided;
        s.photon_reuse = reuse;
        s.manifold_guiding = manifold;
        s.spp = 1;
        r.reset();
        if s.photon_nrc && args.cache_warmup_seconds > 0. {
            let warm = Instant::now();
            while warm.elapsed().as_secs_f64() < args.cache_warmup_seconds {
                r.accumulate(&s);
                wait(&r)?;
            }
            eprintln!(
                "SPPC warmup: {} rounds, {:.2}s",
                r.photon_nrc_batches(),
                warm.elapsed().as_secs_f64()
            );
            r.clear_accumulation();
        }
        let start = Instant::now();
        loop {
            s.spp = if enabled || s.photon_nrc { 1 } else { 8 };
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
        args.output.join("cache-config.txt"),
        format!(
            "SPPC photons={} queries={} hash_log2={} excluded_warmup_seconds={}\n",
            args.photon_nrc_photons,
            args.photon_nrc_queries,
            args.photon_nrc_hash_log2,
            args.cache_warmup_seconds
        ),
    )?;
    std::fs::write(
        args.output.join("configuration.txt"),
        format!(
            "{}x{}; {:?}; depth 20; raw linear; denoiser off; SPPC cache only for photon-nrc; {} spp camera reference ray seed 1347; geometry seed 42; test ray seed 42; warm pipelines excluded, all GPU work waited; imported references use a one-sample primary-normal floor mask; target {}s. NRMSE uses a finite noisy reference, not ground truth.\n",
            args.width, args.height, args.world, args.reference_samples, args.seconds
        ),
    )?;
    Ok(())
}
