//! Reproducible image captures for the project gallery. These use the same GPU
//! kernels and PNG export as the interactive application, without UI chrome.
use anyhow::Result;
use clap::Parser;
use multiworld_path_tracer::{
    gpu::{self, Renderer},
    settings::{Mode, Settings, View},
};
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 800)]
    width: u32,
    #[arg(long, default_value_t = 600)]
    height: u32,
    #[arg(long, default_value_t = 512)]
    samples: u32,
    #[arg(long, default_value = "output/gallery")]
    output: PathBuf,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let (device, queue) = pollster::block_on(gpu::headless_device())?;
    let mut renderer = Renderer::new(device, queue, args.width, args.height, 42)?;
    let base = Settings {
        exposure: 1.,
        spp: 8,
        ..Default::default()
    };
    let cases = vec![
        (
            "spheres",
            Settings {
                mode: Mode::WorldA,
                ..base.clone()
            },
        ),
        (
            "cubes",
            Settings {
                mode: Mode::WorldB,
                ..base.clone()
            },
        ),
        (
            "blend",
            Settings {
                mode: Mode::Blend,
                ..base.clone()
            },
        ),
        (
            "per-path",
            Settings {
                mode: Mode::PerPath,
                ..base.clone()
            },
        ),
        (
            "independent",
            Settings {
                mode: Mode::PerBounce,
                ..base.clone()
            },
        ),
        (
            "persistent",
            Settings {
                mode: Mode::Persistent,
                persistence: 0.95,
                ..base.clone()
            },
        ),
        (
            "borrowed-light",
            Settings {
                primary_a: true,
                p: 0.2,
                exposure: 1.4,
                ..base.clone()
            },
        ),
        (
            "spatial",
            Settings {
                mode: Mode::Persistent,
                spatial: true,
                threshold: 0.5,
                softness: 0.16,
                time: 3.,
                exposure: 1.4,
                ..base.clone()
            },
        ),
        (
            "spatial-orbit",
            Settings {
                mode: Mode::Persistent,
                spatial: true,
                threshold: 0.5,
                softness: 0.16,
                time: 3.,
                yaw: 2.2,
                exposure: 1.4,
                ..base.clone()
            },
        ),
        (
            "primary-worlds",
            Settings {
                spatial: true,
                threshold: 0.5,
                softness: 0.16,
                time: 3.,
                view: View::PrimaryWorld,
                ..base.clone()
            },
        ),
        (
            "switch-count",
            Settings {
                view: View::Switches,
                ..base.clone()
            },
        ),
        (
            "normals",
            Settings {
                mode: Mode::WorldA,
                view: View::Normals,
                ..base.clone()
            },
        ),
        (
            "denoised-32",
            Settings {
                denoise: true,
                spp: 1,
                ..base.clone()
            },
        ),
    ];
    std::fs::create_dir_all(&args.output)?;
    let mut metadata = String::from(
        "# Gallery settings\n\nAll images use seed 42. Camera and transport values are recorded below; view is indicated explicitly.\n\n",
    );
    for (name, mut s) in cases {
        let total = if s.denoise { 32 } else { args.samples };
        renderer.reset();
        while renderer.samples < total {
            s.spp = s.spp.min(total - renderer.samples);
            if s.denoise {
                renderer.render(&s, true);
            } else {
                renderer.accumulate(&s);
            }
        }
        renderer.render(&s, false);
        renderer.save_png(&args.output.join(format!("{name}.png")))?;
        if s.denoise {
            renderer.save_raw_png(&args.output.join("raw-32.png"))?;
        }
        metadata.push_str(&format!(
            "- **{name}**: {}×{}, {total} spp, {s:?}\n",
            args.width, args.height
        ));
        eprintln!("Saved {name}: {total} spp");
    }
    std::fs::write(args.output.join("settings.md"), metadata)?;
    Ok(())
}
