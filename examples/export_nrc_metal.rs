//! Export inspectable MSL for the same fused kernels wgpu runs on Metal.
//! The export uses an explicit resource ABI, not wgpu's private binding map.
use anyhow::Result;
use clap::Parser;
use multiworld_path_tracer::settings::Mode;
use naga::{
    ResourceBinding, ShaderStage,
    back::{PipelineConstants, msl, pipeline_constants::process_overrides},
    valid::{Capabilities, ValidationFlags, Validator},
};
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "output/nrc/metal")]
    output: PathBuf,
    #[arg(long, value_enum, default_value = "per-bounce")]
    mode: Mode,
    #[arg(long)]
    denoise: bool,
    /// Export Apple SIMD matrix inference instead of the portable scalar path.
    #[arg(long)]
    cooperative: bool,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let source = [
        if args.cooperative {
            "enable wgpu_cooperative_matrix;"
        } else {
            ""
        },
        concat!(
            concat!(
                include_str!("../src/shaders/trace.wgsl"),
                "\n",
                include_str!("../src/shaders/adaptive.wgsl")
            ),
            "\n",
            include_str!("../src/shaders/caustics_stub.wgsl")
        ),
        include_str!("../src/shaders/nrc.wgsl"),
        if args.cooperative {
            include_str!("../src/shaders/nrc_matrix.wgsl")
        } else {
            include_str!("../src/shaders/nrc_scalar.wgsl")
        },
        if args.cooperative {
            include_str!("../src/shaders/nrc_queued.wgsl")
        } else {
            ""
        },
        include_str!("../src/shaders/denoise.wgsl"),
    ]
    .join("\n");
    let module = naga::front::wgsl::parse_str(&source)?;
    let info = Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module)?;
    let constants: PipelineConstants = [
        ("NRC_COOPERATIVE".into(), 0.),
        ("NRC_QUEUED".into(), u32::from(args.cooperative) as f64),
        ("TRACE_GROUP_Y".into(), 8.),
        ("NRC_ENABLED".into(), 1.),
        ("ACTIVE_MODE".into(), args.mode as u32 as f64),
        ("DENOISE_INPUT".into(), u32::from(args.denoise) as f64),
    ]
    .into_iter()
    .collect();
    std::fs::create_dir_all(&args.output)?;
    for entry in [
        "neural_train_fused",
        "neural_apply_gradients",
        "neural_finish_training",
        "neural_initialize",
        "neural_teach",
        "accumulate",
        "present",
        "denoise",
    ]
    .into_iter()
    .chain(if args.cooperative {
        vec![
            "neural_resolve_queries",
            "neural_finish_guides",
            "neural_resolve_teachers",
        ]
    } else {
        vec![]
    }) {
        let mut constants = constants.clone();
        constants.insert(
            "NRC_TEACHER".into(),
            u32::from(entry == "neural_teach" || entry == "neural_resolve_teachers") as f64,
        );
        constants.insert(
            "NRC_TEACHER_QUEUED".into(),
            u32::from(args.cooperative && entry == "neural_teach") as f64,
        );
        let (module, info) = process_overrides(
            &module,
            &info,
            Some((ShaderStage::Compute, entry)),
            &constants,
        )?;
        let mut resources = msl::EntryPointResources {
            sizes_buffer: Some(13),
            ..Default::default()
        };
        for binding in [0, 1, 2, 3, 5, 6, 7] {
            resources.resources.insert(
                ResourceBinding { group: 0, binding },
                msl::BindTarget {
                    buffer: Some(binding as u8),
                    mutable: [3, 7].contains(&binding),
                    ..Default::default()
                },
            );
        }
        resources.resources.insert(
            ResourceBinding {
                group: 0,
                binding: 4,
            },
            msl::BindTarget {
                texture: Some(0),
                mutable: true,
                ..Default::default()
            },
        );
        for binding in 0..3 {
            resources.resources.insert(
                ResourceBinding { group: 1, binding },
                msl::BindTarget {
                    buffer: Some((binding + 8) as u8),
                    mutable: binding == 1,
                    ..Default::default()
                },
            );
        }
        resources.resources.insert(
            ResourceBinding {
                group: 1,
                binding: 3,
            },
            msl::BindTarget {
                buffer: Some(12),
                mutable: true,
                ..Default::default()
            },
        );
        let mut options = msl::Options {
            lang_version: (2, 4),
            fake_missing_bindings: false,
            ..Default::default()
        };
        options.per_entry_point_map.insert(entry.into(), resources);
        let pipeline = msl::PipelineOptions {
            entry_point: Some((ShaderStage::Compute, entry.into())),
            ..Default::default()
        };
        let (output, translation) = msl::write_string(&module, &info, &options, &pipeline)?;
        for name in translation.entry_point_names {
            eprintln!("{entry}: {}", name?);
        }
        let header = format!(
            "// Generated from src/shaders/*.wgsl by export_nrc_metal.\n// Mode: {:?}; guides: {}; cooperative: {}.\n// Teacher tracing: 4x64 threads; optional bootstrap resolution: 8x32.\n// Training: 128 groups x32 threads, then apply 35x64, then finish 1x1, four updates/frame.\n// Export ABI: group0 buffer bindings map to same Metal slots; texture4 -> texture0;\n// group1 buffers -> slots8/9/10; query queue -> slot12; buffer-size metadata -> slot13.\n// Runtime wgpu may choose a different resource map. Do not edit generated code.\n",
            args.mode, args.denoise, args.cooperative
        );
        std::fs::write(args.output.join(format!("{entry}.metal")), header + &output)?;
    }
    Ok(())
}
