//! Compare raw linear error and completion-timed cost; denoising stays disabled.
use anyhow::{Result, ensure};
use clap::Parser;
use multiworld_path_tracer::{
    gpu::{self, Pixel, Renderer},
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
    #[arg(long, default_value_t = 32)]
    samples: u32,
    #[arg(long, default_value_t = 8192)]
    reference: u32,
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    #[arg(long, default_value = "output/sampling")]
    output: PathBuf,
}
fn wait(r: &Renderer) -> Result<()> {
    r.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(60)),
    })?;
    Ok(())
}
fn render(r: &mut Renderer, s: &Settings, samples: u32) -> Result<(Vec<Pixel>, f64)> {
    r.reset();
    wait(r)?;
    let start = Instant::now();
    let mut s = s.clone();
    while r.samples < samples {
        s.spp = 16.min(samples - r.samples);
        r.accumulate(&s);
        if r.samples.is_multiple_of(256) {
            wait(r)?;
        }
    }
    wait(r)?;
    let time = start.elapsed().as_secs_f64();
    Ok((r.read_pixels()?, time))
}
fn mse(p: &[Pixel], spp: u32, reference: &[Pixel], reference_spp: u32) -> f64 {
    p.iter()
        .zip(reference)
        .flat_map(|(a, b)| {
            (0..3).map(move |c| {
                (f64::from(a.a[c]) / f64::from(spp) - f64::from(b.a[c]) / f64::from(reference_spp))
                    .powi(2)
            })
        })
        .sum::<f64>()
        / (p.len() * 3) as f64
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (1..=16384).contains(&args.samples)
            && (1..=65536).contains(&args.reference)
            && (1..=9).contains(&args.rounds),
        "Invalid budgets/rounds"
    );
    std::fs::create_dir_all(&args.output)?;
    let (device, queue) = pollster::block_on(gpu::headless_device())?;
    let mut r = Renderer::new(device, queue, args.width, args.height, 42)?;
    std::fs::write(
        args.output.join("settings.txt"),
        format!(
            "GPU: {:?}\nResolution: {}x{}\nSamples: {}\nReference: {} spp MIS\nRounds: {} (median, alternating order)\nSeed: 42\nDepth: 10\nDenoise: false\nMSE: raw RGB; finite reference shares the MIS prefix\n",
            r.device.adapter_info(),
            args.width,
            args.height,
            args.samples,
            args.reference,
            args.rounds
        ),
    )?;
    let mut csv = String::from(
        "mode,samples,reference_spp,cosine_ms,mis_ms,cosine_mse,mis_mse,variance_ratio,error_per_time_gain\n",
    );
    for mode in [
        Mode::WorldA,
        Mode::WorldB,
        Mode::PerBounce,
        Mode::Persistent,
    ] {
        let base = Settings {
            mode,
            ..Default::default()
        };
        let mut cosine = base.clone();
        cosine.light_sampling = false;
        render(&mut r, &cosine, 16)?;
        render(&mut r, &base, 16)?;
        let (reference, _) = render(&mut r, &base, args.reference)?;
        r.render(&base, false);
        r.save_png(&args.output.join(format!("{}-reference.png", mode.slug())))?;
        let mut times = [Vec::new(), Vec::new()];
        let mut errors = [0.; 2];
        for round in 0..args.rounds {
            for step in 0..2 {
                let which = (round + step) % 2;
                let settings = if which == 0 { &cosine } else { &base };
                let (image, time) = render(&mut r, settings, args.samples)?;
                times[which].push(time);
                errors[which] = mse(&image, args.samples, &reference, args.reference);
                if round == 0 {
                    r.render(settings, false);
                    r.save_png(&args.output.join(format!(
                        "{}-{}.png",
                        mode.slug(),
                        if which == 0 { "cosine" } else { "mis" }
                    )))?;
                }
            }
        }
        let times = times.map(|mut t| {
            t.sort_by(f64::total_cmp);
            t[t.len() / 2]
        });
        let variance = errors[0] / errors[1];
        let efficiency = variance * times[0] / times[1];
        let row = format!(
            "{},{},{},{:.3},{:.3},{:.8},{:.8},{:.3},{:.3}\n",
            mode.slug(),
            args.samples,
            args.reference,
            times[0] * 1000.,
            times[1] * 1000.,
            errors[0],
            errors[1],
            variance,
            efficiency
        );
        eprint!("{row}");
        csv.push_str(&row);
    }
    std::fs::write(args.output.join("metrics.csv"), csv)?;
    Ok(())
}
