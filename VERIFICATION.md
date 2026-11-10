# Local verification

Verified on an Apple M1 using the Metal backend, November 9, 2026.

- `cargo fmt --check`: passed.
- `cargo clippy --all-targets -- -D warnings`: passed.
- `cargo test`: five CPU tests passed; the hardware-dependent test is explicitly ignored in this default command.
- `cargo test --test gpu_transport -- --ignored --nocapture`: passed on Metal. Covers all mixture endpoints, both persistence endpoints, dispatch partitioning, blend recomposition without sample reset, separate debug accumulators, restoration of generated geometry after custom fixtures, resizing, spatial probability, and primary-A locking.
- Windowed smoke run: `cargo run -- --width 400 --height 300 --smoke-frames 12` exited successfully and produced `output/interactive-smoke.png`.
- Headless comparison: all six modes rendered at 640×480, 512 spp, depth 10, seed 42, p=0.5, s=0.8. PNGs and settings are in `output/comparison/`. Inspected the sphere-world render, per-bounce render and smoke export visually. The native desktop inspection tool could not attach to the Cargo executable, so individual mouse controls have not been manually exercised.

The analytic two-plane GPU comparison produced RGB-channel-averaged linear radiance:

| Estimator | Measured | Analytic expectation |
| --- | ---: | ---: |
| Image-space blend | 0.606667 | 0.606667 |
| Once per path | 0.606460 | 0.606667 |
| Independent per bounce | 0.828095 | 0.828333 |

This checks a cross-world light contribution, not simply a difference in image noise. Render timing in `render-info.txt` includes concurrent verification activity and is not a controlled performance benchmark. Vulkan/DX12 and non-M1 hardware have not been tested locally. Without next-event estimation, the default scenes remain noisy at 512 spp; increase the sample budget for close comparisons.


## BVH and optimization verification — November 10, 2026

The optimized path is the default. The original linear query remains selectable with `--brute-force`; `--force-bvh` disables the small-sphere fallback. Both CPU and hardware-dependent checks were rerun after the optimization changes.

- Eight CPU tests pass, including BVH structure/containment/escape links, deterministic construction, coincident/empty/plane-only scenes and rejected invalid bounds.
- The direct GPU query test makes **356,010 query comparisons** across BVH and optimized linear algorithms. Distance, oriented normal, diffuse/metal properties and emission agree exactly with the original linear reference for the tested rays. Cases include generated seeds, custom worlds with unequal counts, inside origins from either world, signed zero, tangency, coincident surfaces and a near-parallel thin box.
- Paired renders cover every transport mode, two seeds and four probability/persistence/spatial configurations each, comparing all radiance and debug sums with an absolute tolerance of 1e-6. Both forced and automatic traversal are exercised.
- Fused interactive display and deferred headless display produce identical PNG bytes. View changes, repeated paused frames and a reset while paused preserve or clear sums as appropriate.
- The existing probability endpoints, persistence endpoints, seed restoration, resize, primary-A lock and analytic two-plane checks still pass with acceleration enabled. The analytic means remain the values reported above.
- `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` pass.

No fast-math flags, reduced sample budgets, denoising or altered light-transport distributions are used. One attempted winner-normal shortcut was rejected because strict GPU tests detected a few-ULP difference; the winning surface still uses the reference normal calculation. These checks are evidence for the covered inputs, not a proof for arbitrary geometry or every GPU backend.

See `PERFORMANCE.md` for the final warmed throughput measurements and release invocation. The November 9 outputs remain available as historical comparison images.

## Release 0.2.0 follow-up checks

- Horizontal orbit now uses an unlimited degree control and canonical wrapping;
  the CPU regression crosses both ±π boundaries and exercises repeated turns.
- A minimal GPU repro looking at the same plane point from 24 camera positions
  previously varied from 44 to 1,024 A selections per 1,024 seeds. Linear and BVH
  queries reproduced the same fault; constant probability did not. Surface-
  anchored primary noise now returns 590 at every position, while still varying
  across different surface points. Later segments retain ray-origin noise.
- Denoising tests cover every mode's raw-sample preservation, finite history,
  stationary sample accounting, repeated paused frames, motion reuse/capping,
  probability invalidation, resizing and zero-strength equality with raw PNGs.
  Corrupting only B's geometry guide rejects history even when A remains valid.
- The completed suite has 9 CPU tests and 5 hardware-dependent GPU tests. All
  passed on Apple M1 / Metal; formatting and warning-free Clippy also passed.
- The bundled 0.2.0 application was opened through the native UI. Enabling
  denoising updated the image and preserved accumulated raw samples. The initial
  Finder-launch export check exposed read-only `/` as the process working
  directory; exports now default to a user-owned Pictures folder in that case.
- The distributed binary is rebuilt with home-directory path remapping and
  inspected for local home paths before publication. Source/data captures are
  scanned separately; Git uses a GitHub noreply address.

Denoising is an optional, biased display operation. Its overhead and visual
changes are separate from the unbiased transport and acceleration checks.

The final Finder-launched app was reopened after the export fix. Clicking Save
PNG succeeded in the user-owned Pictures capture directory. Native UI checks
also confirmed the 0.2.0 version label, temporal-denoising toggle/strength,
updated image and raw/display export controls. Public gallery assets are GPU
render captures, not desktop screenshots containing filesystem status text.
