# Apple Silicon acceleration

Implemented and measured on 16 November 2026 using an M5 Max with 40 GPU cores,
64 GB unified memory, macOS 27.2 and the pinned wgpu 30.0.1. The renderer still
uses one wgpu device and retains its portable WGSL implementation.

## Native ray queries

The device requests `EXPERIMENTAL_RAY_QUERY` when available. On Metal it also
requires the Apple9 GPU family or newer, so M1/M2 keep their existing compute
traversal. Each world owns separate native acceleration structures. Every ray,
including a ray that changes worlds or starts inside glass, queries its selected
world from a fresh root. Camera tracing, shadows, glass medium reconstruction,
light-launched paths, adaptive tracing and neural teachers share this interface.

Analytic spheres, boxes, capsules and ellipsoids use procedural bounds and their
existing exact intersection routines. Mesh triangles use thin conservative native
triangle prisms, followed by the original acceptance test. This accommodates
vertex reconstruction roundoff, tessellation seams and nearly coplanar rays
without changing glass normals or the scene-index tie rule. Infinite planes stay
outside the finite acceleration structures. Rays with nonzero direction
components below the reference's `1e-8` parallel-axis threshold use the software
query to preserve that convention.

wgpu 30.0.1's Metal acceleration-structure barrier is a no-op. Building a TLAS before
its BLAS completed produced missing geometry in the regression test. Upload now
waits for the BLAS submission before building its dependent TLAS. This is a
scene-upload boundary, not a per-frame or per-ray wait. The implementation and
capability checks are in [ray_query.rs](../src/gpu/ray_query.rs) and
[Metal's wgpu backend](https://github.com/gfx-rs/wgpu/blob/v30.0.1/wgpu-hal/src/metal/adapter.rs).

Automatic small-scene selection keeps the optimized linear path for at most 32
finite spheres in fixed-world/blend modes. A separate software compute pipeline
specializes away native ray-query code when that path is selected. The UI's
**Hardware ray tracing** checkbox and CLI `--software-tracing` provide comparisons;
`--force-bvh` selects the software BVH and `--brute-force` the linear reference.
Unsupported devices never parse the ray-query extension.

### Completion-timed traversal measurements

512×384, 32 camera samples, three warmed rounds with alternating measurement order.
Times exclude upload, shader compilation, image readback and PNG encoding. Each
row compares the same scene, seed, camera, estimator and sample budget. Sphere
hardware is deliberately forced here to test the automatic-policy decision.

| Workload | Software ms | Native ms | Software / native |
| --- | ---: | ---: | ---: |
| Small sphere world | 12.07 | 17.06 | 0.71× |
| Temple/grove switching | 34.75 | 28.43 | 1.22× |
| Teapot camera paths | 188.65 | 74.27 | 2.54× |
| Teapot with photon caustics | 267.43 | 120.78 | 2.21× |
| Suzanne camera paths | 125.93 | 45.96 | 2.74× |

The automatic policy avoids the sphere slowdown. Native/software teapot camera
and sphere captures were identical. The temple/grove comparison differed at one
pixel (HDR NRMSE `1.98e-6`); Suzanne at three pixels (`2.7e-7`). Photon accumulation
uses floating-point atomics and can vary with GPU execution order; this run's
NRMSE was about `2e-8`. These are finite-sample comparisons, not a guarantee of
bitwise-identical images for every possible scene or a universal speedup.

The benchmark also tested 32-, 64- and 128-thread camera workgroups. The 64-thread
default remains: 32 threads offered no consistent gain and 128 were slower on
the mesh workloads. All four supported workgroup variants, including 256
threads, have a partial-tile/reconstruction-input equivalence check.

## M5 neural TensorOps

Both optional caches have a Metal Performance Primitives `matmul2d` path for
their 32- and 64-wide MLPs. WGSL packs encoded queries and frozen rendering
weights; a native Metal compute pass evaluates both hidden layers with half
inputs and float accumulation; WGSL applies the RGB continuation and updates
the same moments, half images and guides. Invalid query features are not packed;
the native pass masks them without reading stale feature storage. All three passes run in wgpu's ordered
command stream. There is no CPU query readback or per-frame shader compilation.

The native shader has one explicit storage binding and a 128-thread execution
scope matching its TensorOp. Its source is owned by the renderer and compiled
through wgpu's native shader API. The implementation uses the Metal runtime
compiler and does not require the separate offline Metal Toolchain.
[Apple's TensorOps guidance](https://developer.apple.com/videos/play/tech-talks/111432/)
describes the M5 GPU Neural Accelerators targeted by these operations.

Automatic selection requires Apple10 GPU-family support. Older devices retain
SIMD matrix/scalar inference, and compilation/resource-limit failures fall back
to it as well. **M5 tensor inference** in the UI or `--simd-neural` on the CLI
allows comparison. Neural caches themselves remain opt-in. Training and Adam/EMA
updates retain their existing GPU float implementation; TensorOps changes camera
inference, not the learning rule or the cache's approximation boundaries.

The native MLP probe's maximum RGB difference from float reference was
`5.98e-7` at width 32 and `2.96e-6` at width 64. The frozen rendering replay also
checks all six world modes and reconstructed guides. Half inputs can change
predictions; they are not advertised as bitwise-equivalent arithmetic.

A matched 512×384 open-scene comparison, 128 warm-up batches and 32 measured
samples, including ongoing training, measured 43.40 → 40.11 ms in World A and
41.24 → 37.01 ms in per-bounce transport (1.08× and 1.11× throughput). Reported reference-image MSE
was unchanged to eight decimal places. These are single-run comparisons; ordinary
tracing still beats either neural cache on these short open paths.

For the 64-wide photon cache at 512×384, a three-second run after two seconds
of excluded warm-up completed **579 samples with TensorOps versus 449 with SIMD**
(about 29% more samples per second). HDR reference-image NRMSE was `.0521` versus
`.0550`. Both retain ongoing photon tracing and training; equal warm-up time
allows the faster backend to learn more rounds, so this is an end-to-end
comparison rather than a frozen-model kernel speedup.

## Shared Metal OIDN

On Metal, the bounded background worker imports the renderer's native input
allocation into OIDN. wgpu completion hands off an immutable snapshot; OIDN's
separate queue prefilters guides and reconstructs the main and optional half
images asynchronously, then synchronizes once for the batch. A GPU publication
pass copies the current-generation results into the displayed feedback buffer.
The next snapshot cannot reuse storage before publication completes.

This removes mapped-image readback, CPU image conversion and reconstruction
upload from the Metal path. Raw accumulation remains independent. Generation
rejection, blend-world separation, strength blending and export completion retain
their existing behavior. The direct shared/staged test matches exactly, including
signed inputs, both worlds and half-image outputs. Non-Metal devices retain host
staging; `MULTIWORLD_OIDN_STAGING=1` forces it for measurement.

### Equal-time reconstruction comparison

At 512×384 with photon caustics, a two-second run included final current-snapshot
reconstruction. Shared Metal storage completed 492 average camera spp in 2.014 s;
the staged path completed 424 in 2.064 s: about **19% more camera paths per
second**. Adaptive noise allocation improved from 352/2.056 s to 400/2.016 s
(16%); denoised-error allocation improved from 316/2.072 s to 372/2.046 s (19%).
The raw-uniform control completed 516 spp in both runs. All allocations passed
the exact camera-budget check. These are single-run end-to-end measurements,
not isolated OIDN kernel timings. Both use the same validated finite camera
reference; the CSVs retain image-error measurements.

## Large-image adaptive budget correction

The 512×384 reconstruction comparison found an existing allocator rounding error.
At sample index 161, jitter `.9977968` added to the 196,608-ray endpoint rounds to
196,609 in float, spending one extra ray. The scheduler now bins the integer and
fractional parts separately. The regression replays this exact phase and checks
both allocations and executed camera counts across multi-chunk scans.

## Reproduce and verify

```sh
cargo run --release --example acceleration_benchmark -- --output output/m5-acceleration
cargo run --release --example acceleration_benchmark -- --trace-group-y 4 --output output/m5-workgroup-4
cargo run --release --example nrc_benchmark -- --width 512 --height 384 --samples 32 --reference 256 --warmup 128 --output output/m5-nrc-tensor
cargo run --release --example nrc_benchmark -- --width 512 --height 384 --samples 32 --reference 256 --warmup 128 --simd-neural --output output/m5-nrc-simd
cargo run --release --example caustics_benchmark -- --width 512 --height 384 --reference-samples 256 --seconds 3 --photon-nrc --cache-warmup-seconds 2 --only photon-nrc --output output/m5-photon-tensor
cargo run --release --example caustics_benchmark -- --width 512 --height 384 --reference-samples 256 --seconds 3 --photon-nrc --simd-neural --cache-warmup-seconds 2 --only photon-nrc --output output/m5-photon-simd
cargo run --release --example reconstruction_benchmark -- --width 512 --height 384 --reference-samples 256 --seconds 2 --output output/m5-reconstruction-shared
MULTIWORLD_OIDN_STAGING=1 cargo run --release --example reconstruction_benchmark -- --width 512 --height 384 --reference-samples 256 --seconds 2 --output output/m5-reconstruction-staged
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
cargo test --release -- --ignored --nocapture --test-threads=1 --skip neural_gpu_phase_profile
```

GPU tests require actual GPU access; a restricted execution sandbox may hide the
Metal adapter. The full suite covers native/software intersections and transport,
glass, both caches, OIDN, exact adaptive budgets, export, feature-disabled fallback
and alternate workgroups. Large queued SIMD inference also uses two-dimensional
dispatch, so selecting its fallback does not exceed the per-axis workgroup limit.
A 1056×1024 blended-image CLI smoke test exercised 67,584 query workgroups
through this fallback and completed successfully.
Benchmark CSVs are in [benchmarks](benchmarks/).
