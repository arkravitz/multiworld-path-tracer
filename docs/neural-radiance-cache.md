# Experimental neural radiance cache

A diffuse-only adaptation of [Müller et al., *Real-time Neural Radiance Caching for Path Tracing* (SIGGRAPH 2021)](https://research.nvidia.com/labs/rtr/publication/muller2021nrc/). The cache learns while rendering, replacing diffuse scattering continuations with a prediction. It is biased and defaults off. The uncached path tracer is the reference.

## Try it

Enable **Neural radiance cache (experimental)** in the sidebar, or run:

```sh
cargo run --release -- --neural-cache --spp 1
# Opt in to anchored self-training (also available as Shorter training paths):
cargo run --release -- --neural-cache --cache-self-training --spp 1
# Override automatic path-spread termination with a fixed first query bounce:
cargo run --release -- --neural-cache --cache-bounce 1
```

The default `--cache-bounce 0` uses accumulated path area spread against a camera footprint, with the paper's 0.01 threshold. A query requires a secondary diffuse surface and excludes the final depth-limited surface. A positive value instead selects a fixed first query bounce. Mirrors, misses, local emission and finite-depth transport retain their ordinary handling. Prediction starts after at least 512 valid training examples and 32 Adam updates. The sidebar counts rendering/training batches; each batch performs four minibatch updates.

**Reset** clears the image and model. Scene, seed, transport, probability-field, depth and cache-control changes reset the model and optimizer. Camera motion restarts the image while preserving learned world-space lighting. Exposure, pause and display changes do not train. Blend recomposition preserves the model because fixed-world features do not depend on blend weight. Enabling the denoiser preserves the model when its guide allocation grows. `Renderer::clear_accumulation()` restarts only the image.

The display denoiser is independent. **Save raw PNG** exports the unfiltered cached estimator when NRC is enabled; disable NRC for the uncached reference.

## GPU execution

The shaders compile to Metal through wgpu/Naga. There is no CUDA dependency, CPU training, pretrained model or native queue interop.

- **Apple matrix inference:** one 32-thread SIMD group collects up to 32 ray queries, compacts active queries, and evaluates both hidden layers with `coop_mat8x8` operations. These translate to Metal SIMD-group matrix multiply instructions. Activations remain in approximately 8.3 KB of workgroup storage between layers; there are no global intermediate layer tensors or dense-layer dispatches. Traversal writes a GPU query queue, then one coherent inference dispatch evaluates the complete MLP and resolves its radiance into the image. This separates traversal register pressure from neural work; network layers remain fused. Out-of-image lanes participate in barriers; empty query groups skip the network uniformly.
- **Portable inference:** a scalar, per-ray fused MLP remains available when the optional Metal cooperative-matrix feature is unavailable. Ordinary tracing specializes all neural code out. Teacher bookkeeping is also specialized out of rendering kernels.
- **Parallel training:** 128 independent 32-thread groups each fuse one example's forward/backward pass, with about 9.5 KB of shared weights and activations per group. They read frozen training weights and write disjoint gradient records. An ordered reduction/Adam dispatch averages valid gradients, updates the optimizer, and publishes bias-corrected EMA inference weights with decay 0.99. A final dispatch advances counters. Four such minibatches run per rendering dispatch; no group races another group's weight updates.
- **Phase ordering:** prefix traversal → fused query inference → optional guide completion → teacher tracing → optional batched bootstrap resolution → four minibatch updates. Rendering sees immutable EMA weights. Dependent phases use ordered GPU dispatches, without CPU readback or device-wide synchronization inside a kernel. Shared-memory initialization is explicit; redundant backend zero-filling is disabled on these neural pipelines.

The model remains **32 inputs → 32 ReLU → 32 ReLU → 3 linear outputs**, with biases and 2,211 parameters. Inputs include position, triangle-wave position frequencies, normal, albedo, world, remaining depth, mode, local probability and persistence. EMA/training weights, Adam moments, 1,024 records and 128 partial-gradient rows occupy a 1,266,272-byte prefix in the existing guide allocation. The optional query queue uses 128 bytes per pixel at maximum (two blend-world slots) in a separate bind group. Query and denoiser passes use different resources, remaining within the eight-storage-buffer stage limit. Multi-sample NRC calls execute ordered one-sample steps, preserving the requested sample budget; the denoiser consumes each step when displayed. Pipelines compile lazily on first enable.

The network predicts albedo-demodulated diffuse scattered radiance, excluding local emission. At a query, local emission is counted once, then `throughput * albedo * prediction` replaces the entire scattering continuation. Direct lighting is not added twice.

Each dispatch traces 256 separate camera-distributed teacher paths, recording up to four diffuse vertices each. By default they remain fully uncached. With **Shorter training paths (experimental)** / `--cache-self-training`, a ready cache lets most teachers bootstrap at a second diffuse termination event after the normal query cutoff. Exactly 16 of the 256 teachers remain uncached anchors, rotating membership each dispatch. Cold caches always use fully traced teachers. The terminal query does not train on itself; only preceding vertices learn from its transported prediction. Labels reconstruct the suffix after each vertex, removing camera-prefix throughput and the vertex albedo. World and remaining depth accompany every record. Self-training uses frozen raw optimizer weights; rendering uses the separate EMA snapshot. Apple batches the 256 possible teacher queries into the same fused matrix implementation and applies their terminal-to-prefix factors in an ordered resolution pass. The portable fallback evaluates the same prediction directly in the teacher shader. Labels in both paths have an analytic equivalence test. Empty and numerically invalid records are masked. A shuffled permutation supplies four disjoint 128-record minibatches. Relative per-channel squared loss uses a detached prediction denominator; labels/predictions are limited to `[0,64]`, and optimizer gradients are bounded. Those choices can introduce bias in bright or poorly sampled regions.

## What differs from the paper

This revision adopts matrix batching, parallel minibatches, cheap triangle waves, EMA rendering weights and area-spread termination. It does **not** reproduce the paper's complete system: the published network has five 64-wide hidden layers and supports directional/specular materials; its rendering prefixes become training paths with bootstrapped suffixes and a 1/16 uncached fraction. This renderer uses a smaller diffuse model and separate teachers, with optional anchored bootstrapping. Its four minibatches contain 128 records each, versus the paper's 16,384. The small network and teachers remain explicit implementation boundaries, rather than a claim of equivalent quality or RTX performance. [Primary-source research](neural-radiance-cache-research.md).

Online learning makes cached images depend on the training/dispatch schedule. Uncached sample-batch invariance remains intact.

## Inspect the Metal kernels

```sh
# Hardware matrix inference used on supported Apple Metal devices:
cargo run --offline --example export_nrc_metal -- --cooperative
# Portable scalar specialization:
cargo run --offline --example export_nrc_metal
# Different transport mode and guide-producing inference:
cargo run --offline --example export_nrc_metal -- --cooperative --mode persistent --denoise
```

The exporter writes `accumulate.metal`, `neural_train_fused.metal`, `neural_apply_gradients.metal`, `neural_finish_training.metal`, `neural_initialize.metal`, `neural_teach.metal`, `neural_resolve_queries.metal`, `neural_resolve_teachers.metal`, `neural_finish_guides.metal`, `present.metal` and `denoise.metal` under `output/nrc/metal/`. It validates the shaders, resolves overrides and checks entry-point translations. Generated headers describe the explicit inspection ABI; wgpu can choose a different resource map at runtime. WGSL remains authoritative. Naga is only a development dependency.

The actual kernels execute on M1/Metal in the GPU tests. Offline `xcrun metal` additionally needs Apple's separate Metal Toolchain.

## Validation and measurements

```sh
cargo test --release -- --ignored --nocapture --test-threads=1
cargo run --release --example nrc_benchmark
# Explicitly fail if NRC takes longer than ordinary tracing:
cargo run --release --example nrc_benchmark -- --require-faster
# Diagnostic phase costs (run with no other GPU tests/benchmarks):
cargo test --release --lib neural_gpu_phase_profile -- --ignored --nocapture
```

Tests compare hardware matrix predictions against scalar inference, including masked queries and partial image tiles. An integration replay compares queued rendering against scalar inference in all six modes, with and without fresh denoising guides. A world-conditioned supervised fit checks held-out coordinates, with opposite red/blue targets, and achieves raw RGB MSE around 0.000085 after 512 minibatches. Other tests cover emission, mirrors, depth, actual uncached teacher labels, and model preservation/reset. The self-training test checks exact suffix factors in a closed black room, distinct raw/EMA models, exactly 1/16 uncached anchors, scalar/matrix agreement, excluded terminal records, and a fully uncached cold start. The ordinary BVH, transport and denoiser checks run separately.

The benchmark measures raw RGB error against an uncached reference and waits for GPU completion. Timings include inference, teachers and ongoing training; initialization, warm-up, PNG encoding and image readback are excluded. It restarts the measured image while retaining learned weights. The finite reference shares path prefixes with the evaluated images; teacher random streams are independent. Single-run timings are illustrative, not a performance guarantee.

On Apple M1/Metal, the open temple/grove test at 160×120 uses 32 measured spp, 128 warm-up batches and a 256-spp reference:

| Mode | Initial NRC prototype | Optimized NRC | Ordinary tracing | Ordinary / optimized NRC raw MSE |
| --- | ---: | ---: | ---: | ---: |
| World A | 168.20 ms | 101.61 ms | 63.77 ms | 0.001687 / 0.001619 |
| Per bounce | 179.70 ms | 94.27 ms | 47.92 ms | 0.002512 / 0.002316 |

These end-to-end changes make NRC approximately 1.66× and 1.91× faster than the prototype. The prototype used a fixed bounce-2 cutoff; the optimized version uses automatic path spread. Ordinary tracing still wins on these short, open paths.

A second fixture encloses reflective diffuse walls, 25 occluding columns and two small lights, with different column/light placement in the two worlds. It uses 800×600, depth 16, 16 measured spp, 128 warm-up batches and a 256-spp reference:

| Mode | Ordinary tracing | Optimized NRC | Speedup | Ordinary / NRC raw MSE |
| --- | ---: | ---: | ---: | ---: |
| World A | 2,978.94 ms | 634.20 ms | 4.70× | 0.003298 / 0.000273 |
| Per bounce | 845.56 ms | 753.50 ms | 1.12× | 0.197983 / 0.016118 |

The enclosed fixture provides enough long, expensive continuations to amortize neural work. First-surface direct visibility remains traced. The finite reference, especially in mixed transport, is still noisy; lower measured MSE is evidence for this run, not proof of negligible cache bias or temporal stability. Warm-up is excluded from these steady-state timings and must be paid after a scene reset.

```sh
cargo run --release --example nrc_benchmark -- --samples 32 --reference 256 --warmup 128
cargo run --release --example nrc_benchmark -- --enclosed --width 800 --height 600 --samples 16 --reference 256 --warmup 128 --require-faster --output output/nrc/enclosed-800
```

`--cache-bounce 1` selects a fixed earlier cutoff for comparison. Each output folder contains reference, ordinary and cached PNGs, `metrics.csv`, and `configuration.txt`. The enclosed room is a benchmark fixture rather than a preset in the world selector.

The initial serialized prototype was chosen to establish transport and training correctness on Metal. Its scalar inference and sequential training did not implement the paper's throughput strategy. The optimized implementation addresses those costs, with a clear win on the enclosed test and a remaining loss on the open test. The optional cache should therefore be evaluated on the scene and hardware of interest. Larger matrix tiles and half-precision shared tensors were tested on M1 and rejected because they were slower in these workloads.


## SDK-guided self-training

The optional shorter-training-path mode is guided by the public [NVIDIA NRC integration shader](https://github.com/NVIDIA-RTX/NRC/blob/7de30e4eb2a1a27118eb9a6acf5f715d757240c0/Include/Nrc.hlsli) and [settings](https://github.com/NVIDIA-RTX/NRC/blob/7de30e4eb2a1a27118eb9a6acf5f715d757240c0/Include/NrcCommon.h), inspected on 2026-11-14. This is our own WGSL/Metal implementation; it does not link or vendor NVIDIA's runtime. The current SDK documents separate update/query passes, unlike the original paper's shared rendering prefixes. Separate teachers therefore remain a reasonable integration choice.

Self-training trades noisier full path targets for predictions that can carry approximation bias. It is opt-in because the compact model has shown higher error with bootstrapping at a modest warm-up budget; fewer traced teacher vertices do not guarantee a total frame-time improvement. The earlier performance tables above use fully traced teachers.

A single-run Apple M1 comparison uses the enclosed 320×240 fixture, depth 16, 32 measured spp, 128 warm-up batches and a 512-spp uncached reference. Both columns use the final code and include ongoing training:

| Mode | Fully traced teachers | Self-training | Raw MSE, fully traced / self-training |
| --- | ---: | ---: | ---: |
| World A | 258.46 ms | 248.05 ms | 0.00014279 / 0.00024665 |
| Per bounce | 290.66 ms | 289.87 ms | 0.00659966 / 0.00666384 |

The small timing differences do not establish a robust speed advantage. World A's higher error is the reason to preserve fully traced teachers as the default. Additional convergence and temporal-stability evaluation is needed before promoting this option.

```sh
cargo run --release --example nrc_benchmark -- --enclosed --width 320 --height 240 --samples 32 --reference 512 --warmup 128 --output output/nrc/sdk-default
cargo run --release --example nrc_benchmark -- --enclosed --width 320 --height 240 --samples 32 --reference 512 --warmup 128 --self-training --output output/nrc/sdk-self-training
```
