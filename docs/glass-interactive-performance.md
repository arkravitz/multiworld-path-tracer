# Interactive glass performance on Apple M1

Research checked 2026-11-15. These are ranked engineering proposals, not measured speedups. The display correction holds the last completed OIDN snapshot; it must remain intact while optimizing rendering. Preserve the existing equal-time quality benchmark and export strength behavior.

**Implementation update:** the [live display and responsiveness fix](live-display-and-responsiveness.md) now bounds progressive GPU work to one outstanding job and moves interactive export waiting/readback/encoding off the event thread. The measured 24-job burst backlog fell to one; the remaining options below are throughput experiments, not implemented performance guarantees.

## What the implementation already does

`src/gpu/oidn.rs` already runs OIDN on a worker, uses bounded work/readback state, rejects stale generations, and polls without waiting on the UI thread. Normal requests require at least four samples, eight additional samples, and 150 ms since the last request; uncertainty reconstruction has a separate 750 ms cadence. Thus “make denoising asynchronous” is already implemented. The native bridge still stages image data through the CPU and sets filter quality to numeric value 6 (OIDN high quality).

`src/gpu/caustics.rs::prepare` launches photon preparation per rendering dispatch. Auto budget is the next power of two above image area / 4, clamped to 4,096–32,768. Some clear, scan and scatter work uses the full allocated capacity regardless of active photon budget. `src/gpu.rs::dispatch` expands multi-spp caustic work into successive one-sample dispatches. Lazy initialization constructs complete pipeline families when a feature is first enabled. These are concrete profiling targets; none proves the cause of a particular UI stall.

The existing 512×384 benchmark supports learned photon launches, with about 2.4% lower full-image error at equal time. Manifold proposals and weighted pool reuse did not win that comparison; keep them off by default. See [the recorded benchmark](advanced-caustics.md).

## Ranked next steps

| Rank | Change | Expected benefit and cost |
| --- | --- | --- |
| 1 · implemented | Bound outstanding render work; measure input-to-present latency | Prevent obsolete progressive work accumulating ahead of new camera/settings. Small to medium change; throughput and responsiveness must be measured separately. |
| 2 | Render at reduced resolution during interaction, then settle at full resolution | Largest predictable reduction in per-pixel camera and reconstruction work. Medium change; transient preview loses detail. |
| 3 | Separate photon and camera budgets; remove avoidable fixed-capacity work | Direct reduction of expensive glass paths and hash traffic. Medium change; retain estimator normalization and support. |
| 4 | Multi-rate OIDN quality/cadence and less staging | Reduce GPU contention and snapshot latency. Low effort for quality/cadence experiments; native Metal interop is substantially larger. |
| 5 | Profile BVH traversal, specialize hot kernels, then test a compact wavefront prototype | Potential throughput gain, but no guaranteed benefit on M1. High effort for a full scheduler rewrite. |

### 1. Fix scheduling latency before interpreting FPS

Instrument CPU update time, pipeline initialization, CPU encode time, submitted/completed sequence numbers, GPU duration and displayed image age. Record p50/p95 click-to-visible-change latency. A UI repaint can be fast while its rendering sits behind stale GPU submissions. Apple explicitly warns that too much buffered work increases frame latency; its buffering example uses completion callbacks to recycle resources. Avoid waiting for GPU completion on the UI thread. Start by testing one outstanding progressive batch, skipping additional accumulation while it is in flight while still processing UI events. Compare two if the GPU becomes starved. This is a proposed application policy, not a universal optimal queue depth. [Apple buffering guidance](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/TripleBuffering.html)

First-use shader/pipeline compilation is a different source of stalls: move optional pipeline preparation off the event-handling path or prepare the necessary specialized variant ahead of interaction. Apple's renderer guidance covers function specialization and parallel compilation specifically for responsive authoring. Measure cold toggles separately from warmed steady-state performance. [Apple renderer optimization](https://developer.apple.com/videos/play/wwdc2023/10127/)

### 2. Reduced-resolution interaction, full-resolution settling

Try 0.5× and 0.67× dimensions during camera movement or continuous slider edits, returning to full resolution after roughly 200–300 ms idle. Half dimensions means one quarter as many pixels, not a guaranteed 4× total speedup: full-capacity photon passes and fixed work remain. Use a small reusable set of render targets rather than reallocating every pointer event. Reset sample counts and OIDN generation when switching resolution. Keep denoised snapshots matched to their camera and target size; do not mix in raw accumulation deltas.

Begin with spatial upscaling to isolate the work reduction. MetalFX temporal scaling is a later native-integration option requiring depth, motion vectors, jitter and history resets; the current wgpu path needs an interop design. Query device support instead of assuming support from the API name. Refraction and moving caustics warrant separate ghosting checks. [Apple MetalFX descriptor](https://developer.apple.com/documentation/metalfx/mtlfxtemporalscalerdescriptor), [Apple integration reference](https://github.com/apple/game-porting-toolkit/blob/main/game-porting-skills/skills/using-metalfx-temporal-upscaler/references/integration-guide.md)

### 3. Budget the expensive passes independently

Profile camera tracing, photon tracing, clearing, bucket scans, scatter/gather and reconstruction separately. Test 2K/4K/8K photons during interaction before reducing glass path depth: depth truncation can remove physically important transmission/internal reflection. Match clear/scatter ranges to the actual live region where safe; only remove scan work if hash layout and collision handling remain correct. Check single-world modes for unnecessary second-world preparation.

A time governor can choose budgets from recent completed GPU timings, with hysteresis. Do not call the existing spatial adaptive camera sampler an FPS solution: it redistributes a fixed ray budget and previously lost equal-time quality tests. Changing photon count also changes variance and progressive radius behavior, so compare energy and caustic sharpness after equal wall-clock time. Reusing a photon map across camera frames is a larger estimator change because screen splats, per-round averaging, radius progression and guiding epochs currently evolve together; do not merely skip emission and repeatedly average a correlated old map.

### 4. Tune the asynchronous reconstruction already present

Benchmark balanced or fast OIDN quality during motion, retaining high quality for settling/export. Keep filters alive across frames. Schedule from completion time and usefulness of a newer snapshot, preserving the bounded queue and generation checks. OIDN documents quality modes for interactive use, async execution, native Metal command-queue construction and minimizing redundant image copies. A Metal interop path could remove staging, but requires explicit ownership/synchronization and wgpu native-resource access; unified physical memory alone does not eliminate the current copies. [OIDN official API](https://www.openimagedenoise.org/documentation.html)

An async worker does not provide a second GPU: OIDN and path tracing can contend for M1 compute/bandwidth. Measure raw sample throughput and time-to-clean-display together. Reducing denoise frequency may raise spp/s while making displayed snapshots older.

### 5. Treat wavefront scheduling as an experiment

M1 is Apple GPU family 7; hardware-accelerated ray tracing arrived with family 9/M3. The current WGSL software BVH therefore does not acquire hardware traversal by enabling a new shader flag. [Apple GPU families](https://developer.apple.com/documentation/metal/mtlgpufamily), [Apple M3 architecture](https://developer.apple.com/videos/play/tech-talks/111375/)

Wavefront queues can reduce material divergence and worst-case register pressure, while adding queue writes, compaction and dispatch overhead. PBRT explicitly describes the resulting memory-bandwidth tradeoff and fuses some stages. On this small-material software-BVH renderer, first capture occupancy/register and traversal costs. Prototype bounce compaction or a dedicated glass continuation queue, benchmark against the present megakernel, and retain it only for a repeatable equal-quality/equal-time win. [PBRT GPU mapping](https://www.pbr-book.org/4ed/Wavefront_Rendering_on_GPUs/Mapping_Path_Tracing_to_the_GPU)

## Recent research: relevant, but not an immediate M1 upgrade

- **Guided ReSTIR FG+ (2026):** full guided photon resampling for large scenes and many lights. Author implementation uses Falcor and DirectX Raytracing. It is a useful algorithm reference, but is substantially broader than this repository's light-path-pool RIS and has no demonstrated M1 speedup. [Author code and comparison videos](https://github.com/TU-Clausthal-Rendering/Guided-ReSTIR-FG-Plus)
- **ReSTIR PT Enhanced (2026):** reports 2–3× improvement over its ReSTIR PT baseline through reciprocal neighbor selection, footprint-based reconnection and duplication tracking. That is not a speedup over this renderer. Its lesson is that reuse overhead and correlation need engineering; porting the complete estimator is a research project. [NVIDIA paper page](https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/)
- **Ultrafast Screen-Space Refractions and Caustics via Newton's Method (2026):** solves refracted-ray intersections against G-buffer depth in a few iterations with detected failures/fallbacks. Promising for a separate approximate moving preview, but cannot replace full-scene multi-interface glass transport without changing supported effects. [JCGT paper](https://jcgt.org/published/0015/01/03/paper-lowres.pdf)
- **Fast Pseudo Caustics with Ray Differentials (2026):** deterministic, temporally stable approximate refractive shadows/caustics. Evaluate only if an explicitly approximate preview is desirable; preserve the physical progressive path for settling. [Publisher abstract](https://diglib.eg.org/items/16d5d480-410d-4e27-86bc-4050aff201a5), [author implementation](https://github.com/TU-Clausthal-Rendering/FastPseudoCaustics)
- **MetalFX denoising/upscaling (introduced 2025):** potentially combines low-ray reconstruction and upscaling. Availability, required inputs and actual M1 cost must be checked on the target machine; the announcement alone is not evidence of a faster compatible drop-in. [Apple announcement](https://developer.apple.com/videos/play/wwdc2025/205/)

## Acceptance measurements

Use the existing Glass Gallery and absorbing-lens fixtures. Report warmed GPU milliseconds per pass, completed camera spp/s, photons/s, input-to-present p95, denoised snapshot age, memory and 1/4/8-second image errors. Include cold feature toggles and rapid camera/settings changes to expose compile stalls and queue buildup. For moving previews, compare 0.5×/0.67×/1× scale and fixed photon budgets; for settling, verify energy and caustic sharpness against the existing reference. GPU-time improvement, responsive controls, and time to a clean image are separate outcomes.
