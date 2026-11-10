//! Completion-timed, warmed GPU benchmark; excludes setup, PNG encoding and readback.
use anyhow::{Result, ensure};
use clap::Parser;
use multiworld_path_tracer::{
    gpu::{self, Renderer},
    scene,
    settings::{Mode, Settings},
};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 400)]
    width: u32,
    #[arg(long, default_value_t = 300)]
    height: u32,
    #[arg(long, default_value_t = 128)]
    samples: u32,
    #[arg(long, default_value_t = 8)]
    spp: u32,
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    /// Tile finite scene geometry NxN to measure scaling beyond the default scene.
    #[arg(long, default_value_t = 1)]
    tiles: u32,
    #[arg(long, default_value = "output/benchmark.csv")]
    output: PathBuf,
}
fn wait(r: &Renderer) -> Result<()> {
    r.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(60)),
    })?;
    Ok(())
}
fn run(r: &mut Renderer, settings: &Settings, samples: u32, eager: bool) -> Result<f64> {
    r.reset();
    wait(r)?;
    let start = Instant::now();
    let mut s = settings.clone();
    while r.samples < samples {
        s.spp = settings.spp.min(samples - r.samples);
        if eager {
            r.render(&s, true);
        } else {
            r.accumulate(&s);
        }
    }
    r.render(&s, false);
    wait(r)?;
    Ok(start.elapsed().as_secs_f64())
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.rounds > 0
            && args.rounds <= 25
            && args.samples > 0
            && args.samples <= 16384
            && args.spp > 0
            && args.spp <= 64
            && args.tiles > 0
            && args.tiles <= 16,
        "Invalid benchmark dimensions/budget/rounds/tiles"
    );
    let (device, queue) = pollster::block_on(gpu::headless_device())?;
    let mut r = Renderer::new(device, queue, args.width, args.height, 42)?;
    let mut scenes = scene::worlds(42);
    if args.tiles > 1 {
        for world in &mut scenes {
            let source = world.clone();
            world.truncate(1);
            for z in 0..args.tiles {
                for x in 0..args.tiles {
                    let shift_x = (x as f32 - (args.tiles - 1) as f32 * 0.5) * 12.;
                    let shift_z = (z as f32 - (args.tiles - 1) as f32 * 0.5) * 12.;
                    for o in &source[1..] {
                        let mut o = *o;
                        o.center_radius[0] += shift_x;
                        o.center_radius[2] += shift_z;
                        world.push(o);
                    }
                }
            }
        }
        r.set_worlds(&scenes)?;
    }
    let count = scenes[0].len();
    eprintln!(
        "{}x{}, {} spp, {} rounds, {count} objects/world",
        args.width, args.height, args.samples, args.rounds
    );
    let mut csv = String::from(
        "mode,objects_per_world,width,height,samples,spp,round,linear_eager_ms,linear_deferred_ms,forced_bvh_ms,auto_bvh_ms,forced_bvh_speedup,auto_bvh_speedup,total_speedup\n",
    );
    for mode in Mode::ALL {
        let base = Settings {
            mode,
            spp: args.spp,
            distance: 18. * args.tiles as f32,
            ..Default::default()
        };
        let mut linear = base.clone();
        linear.use_bvh = false;
        let mut forced = base.clone();
        forced.adaptive_bvh = false;
        // Exercise all paths before timing and synchronize actual completion.
        run(&mut r, &linear, 8, true)?;
        run(&mut r, &linear, 8, false)?;
        run(&mut r, &base, 8, false)?;
        run(&mut r, &forced, 8, false)?;
        let mut timings = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for round in 0..args.rounds {
            let mut times = [0.; 4];
            // Rotate order across rounds to reduce systematic thermal/order bias.
            for step in 0..4 {
                let which = (round + step) % 4;
                times[which] = match which {
                    0 => run(&mut r, &linear, args.samples, true)?,
                    1 => run(&mut r, &linear, args.samples, false)?,
                    2 => run(&mut r, &forced, args.samples, false)?,
                    _ => run(&mut r, &base, args.samples, false)?,
                };
            }
            for i in 0..4 {
                timings[i].push(times[i]);
            }
            csv.push_str(&format!(
                "{},{count},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}\n",
                mode.slug(),
                args.width,
                args.height,
                args.samples,
                args.spp,
                round + 1,
                times[0] * 1000.,
                times[1] * 1000.,
                times[2] * 1000.,
                times[3] * 1000.,
                times[1] / times[2],
                times[1] / times[3],
                times[0] / times[3]
            ));
        }
        let median = timings.map(|mut t| {
            t.sort_by(f64::total_cmp);
            t[t.len() / 2]
        });
        eprintln!(
            "{}: linear {:.1}ms -> auto BVH {:.1}ms ({:.2}x); forced BVH {:.1}ms; incl. deferred display {:.2}x",
            mode.slug(),
            median[1] * 1000.,
            median[3] * 1000.,
            median[1] / median[3],
            median[2] * 1000.,
            median[0] / median[3]
        );
    }
    if let Some(parent) = args.output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, csv)?;
    eprintln!("Saved {}", args.output.display());
    Ok(())
}
