# Neural radiance caching: research and implementation boundaries

Researched 2026-11-13. Primary reference: Müller, Rousselle, Novák, and Keller, *Real-time Neural Radiance Caching for Path Tracing* (SIGGRAPH 2021). The supplied [YouTube URL](https://www.youtube.com/watch?v=8WEcE-mRoik) could not be retrieved, including its oEmbed metadata; its title and exact contents remain unverified. The method below is grounded in the authors' paper, not an assumed video transcript.

## What the original paper does

The cache predicts scattered radiance, trained online from extended rendering paths. Its 62 inputs are position frequencies (36), spherical-direction blobs (8), spherical-normal blobs (8), roughness blobs (4), and diffuse/specular RGB (6), padded with ones to 64. Five hidden layers have 64 ReLU neurons; explicit biases are omitted. Reflectance factorization multiplies predictions by diffuse plus specular reflectance. Triangle waves and quartic blobs replace expensive primitives. [Paper, Table 1, §§3.6, 5](https://arxiv.org/html/2106.12372v2#S3.SS6)

Termination uses path area spread against a camera footprint (threshold 0.01). Training suffixes bootstrap from the cache, except 1/16 continue with Russian roulette. Four shuffled, disjoint batches of 16,384 records train each frame. Rendering uses bias-corrected EMA weights (0.99), separate from training. Relative squared loss uses a detached predicted-luminance denominator plus 0.01; Adam optimizes weights. [Paper, §§3.2–3.5, 5](https://arxiv.org/html/2106.12372v2#S3.SS2)

Inference fuses layers using shared memory/registers. Training forward/backward passes share that structure, but weight-gradient products use separate CUTLASS GEMMs. This is not one global training/inference kernel. Cache termination trades variance for bias. [Paper, §4](https://arxiv.org/html/2106.12372v2#S4)

## Recommended first implementation (engineering proposal)

These are adaptations for this repository, not claims of reproducing the published implementation or its performance.

- Keep ordinary path tracing as the default and reference. Expose NRC as an optional biased acceleration method, with learning status and reset controls. Preserve the existing denoiser as a separate choice.
- Start with a small fixed-width MLP, explicit position/direction/normal/material features and world conditioning. A smaller network or raw features are valid prototypes but must be described as a simplified NRC variant. Normalize positions using stable scene bounds, not camera bounds.
- Use a conservative diffuse-only query rule after at least one real scattering event. Continue delta/specular chains normally. A fixed bounce threshold is an initial heuristic, not the paper's path-spread termination.
- Define a single cache quantity. Recommended: outgoing scattered radiance excluding local emission. At a cache query, add local emission once and replace the entire subsequent scattering contribution. Do not also add direct illumination already represented by the prediction. Targets must obey the same convention. When reconstructing a target at an earlier vertex, apply only transport after that vertex; never bake the camera-prefix throughput into its label.
- Initially trace training suffixes without cache, using the renderer's ordinary depth/roulette policy, and learn only from those samples. This avoids self-confirming bootstrap errors during bring-up, although a finite depth cap still imposes the baseline's truncation bias. Later add bootstrapped suffixes with a persistent independent uncached fraction.
- Freeze the weights used to generate targets and render a frame. Run bounded minibatch updates afterward. A compact single-workgroup training kernel can fuse forward, backward, gradient reduction, and update because it owns the entire minibatch. This is a deliberately smaller design than high-throughput multi-workgroup training.

## Multiworld correctness requirements

The same position, direction, and material can have different outgoing radiance in different worlds. A world-agnostic model would average incompatible targets.

- Include the current world in every training record and inference query. Separate model banks are simplest; one-hot world inputs are another option.
- The transport law also depends on process mode and settings such as persistence. Reset weights, optimizer moments, sample records, query counts, and accumulated image whenever those settings or scene content change, unless all relevant state is explicitly conditioned.
- Include bounce/remaining-depth information when the transition process depends on depth or the finite horizon materially changes targets. Include additional path state if it changes future world transitions. Current world ID alone is sufficient only for a stationary Markov transport law.
- Reuse spatial coordinates consistently across worlds, including any transforms used by intersection code. Carry the actual world after a transition into the record.
- Camera motion can preserve a world-space model, but image accumulation and screen-space history must reset as usual. Do not silently preserve learned data after scene/light/material replacement.

## Metal and wgpu integration

The authors' [tiny-cuda-nn reference](https://github.com/NVlabs/tiny-cuda-nn) requires CUDA/NVIDIA hardware; it is an architectural reference rather than a usable Apple runtime. A Metal implementation needs its own fused MSL kernel. Fused means retaining intermediate activations on-chip between layers; merely dispatching separate dense layers under one host API does not provide the same property.

Use an optional macOS backend with a clear portable fallback. Prefer a small, isolated interop boundary; wgpu 30 exposes [unsafe backend access](https://docs.rs/wgpu/30.0.1/wgpu/struct.Device.html#method.as_hal), whose ownership and synchronization requirements must be met. Do not access or submit to the underlying Metal queue concurrently with wgpu work without an explicit ordering contract. A correctness-first implementation may use bounded CPU staging and a native Metal command buffer, waiting for completion before uploading results to wgpu; report the transfer cost honestly.

For training, never let multiple threadgroups read and update the same weights in one dispatch. Either let one group own all updates, or emit partial gradients and reduce/update in separate ordered dispatches. A threadgroup barrier cannot synchronize the whole GPU. Ensure every participating thread reaches every barrier; mask inactive lanes instead of returning early. WGSL likewise defines barriers at workgroup scope and requires uniform control flow. [WGSL memory model and uniformity](https://www.w3.org/TR/WGSL/#memory-model)

Keep immutable inference weights while tracing. Publish updated weights only after training completes. On native Metal, order dependent dispatches with appropriate encoder boundaries/resource barriers and complete GPU writes before CPU reads. Apple exposes [compute resource barriers](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/memorybarrier(resources:)) and [CPU/GPU event synchronization](https://developer.apple.com/documentation/metal/synchronizing-events-between-a-gpu-and-the-cpu).

## Validation before performance claims

Use held-out paths with independent random streams and no cache termination; never use the learned model as its own validation reference. Compare linear radiance, per-world errors, and high-sample ordinary-path-traced images at matched transport settings. Test worlds with opposite lighting, persistence extremes, every process mode, scene changes, specular chains, emission accounting, finite outputs, zero/partial minibatches, and cache resets. Include a fixed supervised fit with known targets to distinguish a broken optimizer from transport-target errors. Measure frame time including training, synchronization, and transfer; NVIDIA timing in the paper is not evidence for Apple performance.

## Implemented portable backend and inspecting Metal

The implementation uses WGSL throughout, translated to MSL by wgpu/Naga. The optimized Apple path uses feature-gated cooperative 8×8 matrices, compacted 32-query batches and on-chip hidden activations. Traversal produces a GPU query queue; a separate coherent dispatch fuses all neural layers, then resolves the image. This avoids coupling traversal register pressure to network execution. The portable path remains scalar. Training now emits independent per-example gradients from 128 groups, then reduces them and applies Adam/EMA in ordered dispatches, four minibatches per rendering dispatch. Triangle-wave encoding, EMA decay 0.99, and area-spread termination replace the initial prototype's sinusoidal encoding, direct weight publication and fixed-only cutoff.

The compact 32→32→32→3 diffuse model and separate teachers remain adaptations. Anchored bootstrapped suffixes are now available as an opt-in mode; cold-start and default teachers remain fully traced. Rendering-prefix promotion, the five-layer 64-wide model and the paper's large batch sizes are not implemented. Benchmarks include the remaining teacher and training cost. See [implementation and measurements](neural-radiance-cache.md).

For an inspectable Metal artifact, Naga 30 can parse the concatenated shaders, validate them, resolve `NRC_ENABLED=true` and `ACTIVE_MODE`, and emit MSL with `back::msl::write_string`. Resolve overrides before writing: `back::pipeline_constants::process_overrides` accepts `Some((ShaderStage::Compute, entry_name))` and a `PipelineConstants` map, returning a specialized module and validation info. Supply explicit per-entry-point resource maps, set `fake_missing_bindings=false`, and inspect `TranslationInfo.entry_point_names` for translation errors. Exporting with an explicit resource map produces valid inspectable kernel source, but does not claim byte-identical output to wgpu's private pipeline compilation options. These APIs were checked against installed Naga 30.0.1 source.


## Current NVIDIA SDK findings (2026-11-14)

Inspected NRC v0.16 headers at commit `7de30e4eb2a1a27118eb9a6acf5f715d757240c0` and the [current integration guide](https://github.com/NVIDIA-RTX/RTXGI/blob/main/Docs/NrcGuide.md). These are source/document observations, not recovered binary internals.

- The guide specifies independent lower-resolution update tracing, full-resolution query tracing, then neural inference/resolution. This differs from the 2021 paper's rendering-prefix promotion and supports retaining our independent teacher pass.
- Public `Nrc.hlsli` implements a second spread-triggered cutoff for training paths and marks an uncached fraction. `NrcCommon.h` exposes a default uncached fraction of 0.0625, radiance-scale controls, scene bounds, feature size, training budget, and self-training attenuation. These are useful integration references; the shader glue does not expose the internal network implementation.
- The distributed binaries target D3D12/Vulkan and depend on CUDA; the arm64 package consists of Windows DLLs, not Metal libraries. Binary kernel recovery would still require an independent Apple GPU implementation, so it is not the chosen engineering path.
- Our optional self-training implementation uses a second diffuse cutoff, rotating 1/16 uncached anchors, frozen raw optimizer weights for teacher predictions, and a separate EMA for rendering. The terminal query is excluded from its own supervision. Apple resolves training queries in fused 32-query matrix groups; the fallback predicts directly. Both paths preserve world/remaining-depth inputs and the renderer's finite-depth target.
- Scene/radiance normalization, a training-record budget that adapts to valid yield, richer directional features, and compact queue records remain candidates for later work. They have not been ported in this revision.

Sources: [integration shader](https://github.com/NVIDIA-RTX/NRC/blob/7de30e4eb2a1a27118eb9a6acf5f715d757240c0/Include/Nrc.hlsli), [public settings](https://github.com/NVIDIA-RTX/NRC/blob/7de30e4eb2a1a27118eb9a6acf5f715d757240c0/Include/NrcCommon.h), [structures](https://github.com/NVIDIA-RTX/NRC/blob/7de30e4eb2a1a27118eb9a6acf5f715d757240c0/Include/NrcStructures.h).
