//! Paired, completion-timed traversal benchmark on the actual selected device.
use anyhow::{Result, ensure};
use clap::Parser;
use multiworld_path_tracer::{
    gpu::{self, Renderer},
    scene::WorldPreset,
    settings::{Mode, Settings},
};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 512)]
    width: u32,
    #[arg(long, default_value_t = 384)]
    height: u32,
    #[arg(long, default_value_t = 32)]
    samples: u32,
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    #[arg(long, default_value_t = 8)]
    trace_group_y: u32,
    #[arg(long, default_value = "output/m5-acceleration")]
    output: PathBuf,
}
fn wait(r: &Renderer) -> Result<()> {
    r.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(60)),
    })?;
    Ok(())
}
fn run(r: &mut Renderer, s: &Settings, samples: u32) -> Result<f64> {
    r.reset();
    wait(r)?;
    let start = Instant::now();
    while r.samples < samples {
        r.accumulate(&Settings {
            spp: (samples - r.samples).min(4),
            ..s.clone()
        });
    }
    wait(r)?;
    Ok(start.elapsed().as_secs_f64() * 1000.)
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        a.samples > 0 && a.samples <= 16384 && a.rounds > 0 && a.rounds <= 25,
        "Invalid sample/round budget"
    );
    let (device, queue) = pollster::block_on(gpu::headless_device())?;
    let mut r =
        Renderer::new_with_workgroup(device, queue, a.width, a.height, 42, a.trace_group_y)?;
    ensure!(
        r.hardware_ray_tracing_available(),
        "Hardware ray queries are unavailable on this adapter"
    );
    std::fs::create_dir_all(&a.output)?;
    let mut csv =
        String::from("scene,width,height,samples,round,software_ms,hardware_ms,speedup\n");
    for (name, worlds, mode, depth, caustics) in [
        (
            "spheres",
            [WorldPreset::WarmSpheres, WorldPreset::CoolBoxes],
            Mode::WorldA,
            10,
            false,
        ),
        (
            "multiworld",
            [WorldPreset::SunkenTemple, WorldPreset::BioluminescentGrove],
            Mode::PerBounce,
            12,
            false,
        ),
        (
            "glass-camera",
            [WorldPreset::GlassGallery, WorldPreset::CrystalGarden],
            Mode::WorldA,
            20,
            false,
        ),
        (
            "glass-photons",
            [WorldPreset::GlassGallery, WorldPreset::CrystalGarden],
            Mode::WorldA,
            20,
            true,
        ),
        (
            "suzanne",
            [WorldPreset::GlassGallery, WorldPreset::CrystalGarden],
            Mode::WorldB,
            20,
            false,
        ),
    ] {
        let base = Settings {
            worlds,
            mode,
            depth,
            caustics,
            photon_guiding: caustics,
            adaptive_bvh: false,
            yaw: 0.6,
            pitch: 0.55,
            distance: 13.,
            ..Default::default()
        };
        let software = Settings {
            adaptive_bvh: true,
            hardware_ray_tracing: false,
            ..base.clone()
        };
        for s in [&software, &base] {
            run(&mut r, s, 8)?;
        }
        let mut times = [Vec::new(), Vec::new()];
        for round in 0..a.rounds {
            let mut t = [0.; 2];
            for i in 0..2 {
                let path = (round + i) % 2;
                t[path] = run(&mut r, if path == 0 { &software } else { &base }, a.samples)?;
            }
            csv.push_str(&format!(
                "{name},{},{},{},{round},{:.4},{:.4},{:.4}\n",
                a.width,
                a.height,
                a.samples,
                t[0],
                t[1],
                t[0] / t[1]
            ));
            for i in 0..2 {
                times[i].push(t[i]);
            }
        }
        for t in &mut times {
            t.sort_by(f64::total_cmp);
        }
        eprintln!(
            "{name}: software {:.2} ms, hardware {:.2} ms, {:.2}x",
            times[0][a.rounds / 2],
            times[1][a.rounds / 2],
            times[0][a.rounds / 2] / times[1][a.rounds / 2]
        );
        // Same scene, seed, sample budget and estimator. Save both raw results
        // alongside the timings for visual review; PNG work is outside timing.
        let mut images = Vec::new();
        for (label, s) in [("software", &software), ("hardware", &base)] {
            run(&mut r, s, a.samples)?;
            images.push(r.read_linear_image(s, false)?);
            r.save_raw_png(&a.output.join(format!("{name}-{label}.png")))?;
        }
        let mut squared = 0_f64;
        let mut energy = 0_f64;
        let mut different = 0usize;
        let mut max_error = 0_f32;
        for (a, b) in images[0].iter().zip(&images[1]) {
            if a != b {
                different += 1;
            }
            for c in 0..3 {
                let d = a[c] - b[c];
                squared += (d as f64).powi(2);
                energy += (a[c] as f64).powi(2);
                max_error = max_error.max(d.abs());
            }
        }
        eprintln!(
            "{name} image comparison: {different} differing pixels, NRMSE {:.8}, max RGB error {max_error}",
            (squared / energy.max(1e-30)).sqrt()
        );
        std::fs::write(
            a.output.join(format!("{name}-comparison.txt")),
            format!(
                "differing_pixels={different}\nnrmse={}\nmax_rgb_error={max_error}\n",
                (squared / energy.max(1e-30)).sqrt()
            ),
        )?;
    }
    std::fs::write(a.output.join("timings.csv"), csv)?;
    Ok(())
}
