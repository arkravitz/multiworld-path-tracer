//! Uncached high-spp reference, raw RGB error, and GPU-completion timings.
use anyhow::{Result, ensure};
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
    #[arg(long, default_value_t = 160)]
    width: u32,
    #[arg(long, default_value_t = 120)]
    height: u32,
    #[arg(long, default_value_t = 64)]
    samples: u32,
    #[arg(long, default_value_t = 2048)]
    reference: u32,
    #[arg(long, default_value_t = 256)]
    warmup: u32,
    #[arg(long, default_value = "output/nrc/comparison")]
    output: PathBuf,
    /// Fail when the neural mode takes longer than ordinary tracing.
    #[arg(long)]
    require_faster: bool,
    #[arg(long, default_value_t = 0)]
    cache_bounce: u32,
    /// Closed, reflective diffuse room with occluding columns and small lights.
    #[arg(long)]
    enclosed: bool,
    /// Enable anchored bootstrapped training for the cached estimator.
    #[arg(long)]
    self_training: bool,
    /// Compare the previous SIMD matrix path against native TensorOps.
    #[arg(long)]
    simd_neural: bool,
}
fn wait(r: &Renderer) -> Result<()> {
    r.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(60)),
    })?;
    Ok(())
}
fn run(r: &mut Renderer, s: &Settings, samples: u32) -> Result<(Vec<Pixel>, f64)> {
    r.clear_accumulation();
    wait(r)?;
    let start = Instant::now();
    for _ in 0..samples {
        r.accumulate(s);
        if r.samples.is_multiple_of(64) {
            wait(r)?;
        }
    }
    wait(r)?;
    let seconds = start.elapsed().as_secs_f64();
    Ok((r.read_pixels()?, seconds))
}
fn error(a: &[Pixel], n: u32, b: &[Pixel], m: u32) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(a, b)| {
            (0..3).map(move |c| {
                (f64::from(a.a[c]) / f64::from(n) - f64::from(b.a[c]) / f64::from(m)).powi(2)
            })
        })
        .sum::<f64>()
        / (a.len() * 3) as f64
}
fn enclosed_worlds() -> [Vec<multiworld_path_tracer::scene::Object>; 2] {
    use multiworld_path_tracer::scene::Object;
    let block = |p: [f32; 3], half: [f32; 3], color: [f32; 3]| Object {
        center_radius: [p[0], p[1], p[2], 0.],
        extent_shape: [half[0], half[1], half[2], 1.],
        color_material: [color[0], color[1], color[2], 0.],
        emission: [0.; 4],
    };
    std::array::from_fn(|world| {
        let mut objects = vec![Object {
            extent_shape: [0., 0., 0., 2.],
            ..block([0.; 3], [0.; 3], [0.82; 3])
        }];
        for (p, half) in [
            ([-10., 4.5, 0.], [0.2, 4.5, 10.]),
            ([10., 4.5, 0.], [0.2, 4.5, 10.]),
            ([0., 4.5, -10.], [10., 4.5, 0.2]),
            ([0., 4.5, 10.], [10., 4.5, 0.2]),
            ([0., 9., 0.], [10., 0.2, 10.]),
        ] {
            objects.push(block(p, half, [0.84; 3]));
        }
        for z in 0..5 {
            for x in 0..5 {
                let color = if world == 0 {
                    [0.82, 0.67, 0.46]
                } else {
                    [0.46, 0.72, 0.82]
                };
                let shift = if world == 0 { 0. } else { 0.65 };
                objects.push(block(
                    [(x as f32 - 2.) * 2.2 + shift, 2., (z as f32 - 2.) * 2.2],
                    [0.32, 2., 0.55],
                    color,
                ));
            }
        }
        for x in [-3., 3.] {
            objects.push(Object {
                center_radius: [x, 7.5, if world == 0 { -3. } else { 3. }, 0.45],
                extent_shape: [0.; 4],
                color_material: [0.; 4],
                emission: if world == 0 {
                    [22., 14., 7., 0.]
                } else {
                    [7., 16., 22., 0.]
                },
            });
        }
        objects
    })
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.samples > 0
            && args.samples <= 4096
            && args.reference > 0
            && args.reference <= 16384
            && args.warmup > 0
            && args.warmup <= 8192,
        "Invalid budgets"
    );
    let (device, queue) = pollster::block_on(gpu::headless_device())?;
    let mut r = Renderer::new(device, queue, args.width, args.height, 42)?;
    std::fs::create_dir_all(&args.output)?;
    ensure!(args.cache_bounce <= 63, "Invalid cache bounce");
    std::fs::write(
        args.output.join("configuration.txt"),
        format!(
            "{}x{}; enclosed={}; cache_bounce={}; self_training={}\n",
            args.width, args.height, args.enclosed, args.cache_bounce, args.self_training
        ),
    )?;
    let mut csv = String::from(
        "mode,raw_spp,warmup_batches,reference_spp,reference_ms,path_ms,nrc_ms,path_mse,nrc_mse\n",
    );
    let mut slower = false;
    for mode in [Mode::WorldA, Mode::PerBounce] {
        let base = Settings {
            worlds: [WorldPreset::SunkenTemple, WorldPreset::BioluminescentGrove],
            mode,
            tensor_neural: !args.simd_neural,
            spp: 1,
            depth: if args.enclosed { 16 } else { 10 },
            distance: if args.enclosed { 8. } else { 18. },
            pitch: if args.enclosed { 0.18 } else { 0.38 },
            ..Default::default()
        };
        run(&mut r, &base, 1)?;
        if args.enclosed {
            r.set_worlds(&enclosed_worlds())?;
        }
        let (reference, reference_time) = run(&mut r, &base, args.reference)?;
        r.render(&base, false);
        r.save_png(&args.output.join(format!("{}-reference.png", mode.slug())))?;
        let (ordinary, ordinary_time) = run(&mut r, &base, args.samples)?;
        r.render(&base, false);
        r.save_png(&args.output.join(format!("{}-path.png", mode.slug())))?;
        let cache = Settings {
            neural_cache: true,
            cache_bounce: args.cache_bounce,
            cache_self_training: args.self_training,
            ..base.clone()
        };
        // Online training and prediction use independent teacher random streams.
        // The subsequent measurement starts a fresh image without forgetting weights.
        r.reset();
        run(&mut r, &cache, args.warmup)?;
        let (neural, neural_time) = run(&mut r, &cache, args.samples)?;
        r.render(&cache, false);
        r.save_png(&args.output.join(format!("{}-neural.png", mode.slug())))?;
        let row = format!(
            "{},{},{},{},{:.3},{:.3},{:.3},{:.8},{:.8}\n",
            mode.slug(),
            args.samples,
            args.warmup,
            args.reference,
            reference_time * 1000.,
            ordinary_time * 1000.,
            neural_time * 1000.,
            error(&ordinary, args.samples, &reference, args.reference),
            error(&neural, args.samples, &reference, args.reference)
        );
        slower |= neural_time >= ordinary_time;
        eprint!("{row}");
        csv.push_str(&row);
        std::fs::write(args.output.join("metrics.csv"), &csv)?;
    }
    ensure!(
        !args.require_faster || !slower,
        "Neural cache is slower than ordinary tracing"
    );
    Ok(())
}
