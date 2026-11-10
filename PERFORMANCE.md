# Rendering performance and release build

Measured on Apple M1 / Metal, November 10, 2026. Release build, seed 42, depth 10, p=0.5, s=0.8. Three warmed rounds with rotating order and explicit GPU completion waits. No concurrent benchmark render jobs were launched. Normal desktop activity and thermal variation can affect timings.

The reference uses the original primitive/linear intersection loop, with the common transport-mode specialization also enabled. The optimized path includes candidate-distance queries and automatic per-world BVH selection. Both columns defer display until the final sample. These ratios isolate query improvements; they do not claim to compare every change against the November 9 executable.

## Default scene: 29 objects per world

400×300, 128 spp, 8 samples per dispatch. Blend spends that budget in each world. Times are medians of three rounds; speedup is the ratio of median times.

| Mode | Linear reference (ms) | Automatic acceleration (ms) | Speedup |
| --- | ---: | ---: | ---: |
| 01-world-a | 188.0 | 123.8 | 1.52× |
| 02-world-b | 675.2 | 305.0 | 2.21× |
| 03-blend | 946.5 | 476.1 | 1.99× |
| 04-per-path | 766.3 | 327.3 | 2.34× |
| 05-per-bounce | 753.0 | 355.5 | 2.12× |
| 06-persistent | 770.6 | 356.9 | 2.16× |

Small sphere-only worlds use the optimized linear candidate loop in fixed-world modes. Forcing BVH on that tiny scene is slower; the automatic policy avoids this case. Stochastic modes use BVH for both worlds to avoid divergent traversal algorithms within the same SIMD group.

## Larger scene: 253 objects per world

320×240, 64 spp, 8 samples per dispatch, 3×3 tiled finite geometry. Camera distance scales with tile count. This throughput capture preceded the later spatial-primary and denoising additions; spatial noise and denoising were disabled. It demonstrates BVH scaling rather than a final-version denoising benchmark.

| Mode | Linear reference (ms) | Automatic acceleration (ms) | Speedup |
| --- | ---: | ---: | ---: |
| 01-world-a | 477.4 | 118.7 | 4.02× |
| 02-world-b | 1982.1 | 168.3 | 11.77× |
| 03-blend | 3795.3 | 563.3 | 6.74× |
| 04-per-path | 3454.2 | 328.2 | 10.52× |
| 05-per-bounce | 3302.8 | 323.1 | 10.22× |
| 06-persistent | 3412.7 | 326.5 | 10.45× |

## Release executable

```sh
./target/release/multiworld-path-tracer
```

The executable is a macOS ARM64 Mach-O binary. Run it directly; Cargo is not needed to launch the built application. Build it again with `cargo build --release`.

## Reproduce

```sh
cargo run --release --example benchmark -- --width 400 --height 300 \
  --samples 128 --spp 8 --rounds 3 --output output/benchmark-default.csv
cargo run --release --example benchmark -- --tiles 3 --width 320 --height 240 \
  --samples 64 --spp 8 --rounds 3 --output output/benchmark-large.csv
```

The CSV records every round for the linear reference with per-batch display, deferred linear reference, forced BVH, and automatic acceleration. Construction, shader compilation, PNG encoding and readback are excluded. GPU wall-clock completion time is measured rather than hardware timestamp queries. Timings have ordinary system/thermal variability; Vulkan/DX12 are not locally benchmarked.

## Correctness checks

See `VERIFICATION.md`. The original reference remains available via `--brute-force`, and `--force-bvh` bypasses the automatic fallback. Sampling budgets, random streams, throughput, Russian roulette, tone mapping and debug statistics are preserved.
