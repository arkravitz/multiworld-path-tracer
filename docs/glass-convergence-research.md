# Faster glass convergence: next steps

Research cutoff: **15 November 2026**.

**Implementation update:** learned light/caster launches, analytic sphere manifold-guided emission, weighted reservoir reuse and independent photon budgets are now available. [Implementation and all-combination measurements](advanced-caustics.md) distinguish these scoped methods from the full paper architectures. Learned launches won the local teapot comparison and are enabled in the glass default; manifold guiding and reuse remain off. The rankings below record the research recommendation before that implementation. This targeted primary-source review extends [the earlier survey](caustics-research.md) after reading [the implemented photon estimator](fast-caustics.md) and [OIDN/adaptive sampling](reconstruction-and-adaptive-sampling.md). Rankings below are engineering judgments for this renderer, not measured speedups.

**Recommendation:** first improve useful photons per millisecond with feedback-guided emission and an independently adjustable photon budget. Keep exact visible splats and the existing transport partition. Profile before replacing gathering or adding reservoirs. For sharp caustics seen through glass, a narrowly scoped manifold next-event estimator is the more ambitious next experiment; full VCM or ReSTIR FG+ is a separate renderer project.

## What is already solved, and what remains

This renderer already follows light through specular chains, splats directly visible diffuse receivers without a density radius, and gathers at receivers reached through glass or additional camera bounces. Recommending basic photon mapping again would duplicate existing work. Its local measurements show that extra light passes improve floor error but reduce camera throughput, while uniform OIDN beats both adaptive camera modes in the documented short M1 comparison. These are scene-specific observations from the linked local benchmark notes.

Visible-splat speckle is primarily a sampling problem: better emission proposals or more useful photons can reduce it without spatial blur. Gathered caustics have an additional bandwidth problem: a larger radius reduces grain while blurring detail; fewer photons with a smaller radius increase variance. OIDN can improve perceived quality, but reconstruction cannot establish an unsampled caustic's exact shape. Neither unbiased Monte Carlo nor consistent progressive photon mapping guarantees a perfect finite-time image; consistency concerns the infinite-sample limit. [SPPM implementation and derivation](https://www.pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Stochastic_Progressive_Photon_Mapping)

## Ranked upgrades

### Local diagnostic, 15 November 2026

Re-ran the existing `reconstruction_benchmark` with the cached independent 1,024-spp reference, 512×384, a two-second target and strength 0.8. Completion includes the final current-snapshot reconstruction, so these short jobs overrun their targets. These are single runs on the local M1, not new general speedup claims.

| Configuration | Camera spp | Completed seconds | Display RMSE |
| --- | ---: | ---: | ---: |
| Fast caustics, uniform OIDN | 12 | 2.82 | 0.03546 |
| Fast caustics, denoising-aware OIDN | 12 | 3.66 | 0.03546 |
| Camera-only caustics, uniform OIDN | 16 | 2.74 | 0.03863 |

At 12 spp the adaptive allocator has not passed its 16-spp uniform warmup. Extra split reconstructions therefore add cost before they can improve camera allocation. Disabling photons increases camera throughput in this scene but worsens displayed error in the comparable short OIDN runs. This supports uniform OIDN as the glass workspace default and motivates improving useful photon work rather than simply removing it. It does not isolate all remaining noise sources; region-wise raw-error and per-pass profiling are still needed before implementing the proposed guide.

```sh
target/release/examples/reconstruction_benchmark --width 512 --height 384 \
  --reference-samples 1024 --seconds 2 --reuse-reference \
  --output output/reconstruction-benchmark-512
# Repeat with --no-caustics to change only the caustic estimator.
```

### Priorities

| Rank | Upgrade | Benefit and scope | Work / main risk |
|---|---|---|---|
| **1** | Feedback-guided photon emission; tune photon/camera budget separately | Spend more emission work on light–caster combinations and directions whose landings actually contribute. Improves both splats and gathering without deliberately broadening either. | Moderate. Preserve sampling support and exact mixture PDFs; large area lights need emission-position-dependent guidance. |
| **2** | Improve gather bandwidth only where gathered transport dominates error | Local density-aware bandwidth, surface compatibility and a progressive accumulation design can improve the blur/noise balance. | Moderate to substantial. The current global shrinking radius is already progressive; canonical SPPM is a different state update, not just a new radius constant. |
| **3** | Restricted manifold next-event estimation for analytic smooth glass | Find a physically bent connection from a receiver to a light; target sharp caustics visible through another glass surface. | Large. Robust roots, visibility, derivatives, discrete reflection/refraction chains and probability estimation. Prototype one sphere or slab before triangle meshes. |
| **4** | ReSTIR caustic reuse / Guided ReSTIR FG+ | Reuse rare successful paths across pixels and time; valuable for large scenes and moving views. | Very large. New reservoir/path state, valid shift mappings/Jacobians, visibility and history rejection. |
| **5** | General BDPT/VCM | Broad transport robustness through multiple path construction strategies. | Very large. Forward/reverse PDFs and MIS across connections/merges; diffuse–specular–diffuse and more complicated chains require careful strategy coverage. |

### 1. Guidance is the smallest new estimator improvement

[Online Photon Guiding with 3D Gaussians (2024 preprint, revised August)](https://arxiv.org/html/2403.03641v2) learns per-light spatial targets from useful gathered photons, adds adaptive light selection and derives a directional sampling transform. It specifically identifies area-light parallax as a weakness of direction-only histograms. For this implementation, start simpler: retain the current caster cones and unguided component, collect contribution statistics by light/caster, and update next-round selection probabilities. Later subdivide useful cones or use spatial targets conditional on the emitter point. This staged approach is our recommendation, not that paper's full algorithm.

Training must use prior data; weight samples by their actual proposal and normalize by launched paths, including misses. Keep nonzero exploration for offscreen casters and unseen chains. Record original proposal labels or probabilities in photon records if feedback needs them. Count useful visible and gathered contributions separately: a receiver hidden from the pinhole may still matter through glass. A view-only success test would wrongly starve those paths.

### 2. SPPM helps density estimation; it does not remove kernel bias instantly

[Progressive Photon Mapping (2008)](https://graphics.ucsd.edu/~henrik/papers/progressive_photon_mapping/) and [SPPM (2009)](https://graphics.ucsd.edu/~henrik/papers/sppm/) trade decreasing bandwidth against increasing accumulated evidence. The [executable pbrt formulation](https://www.pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Stochastic_Progressive_Photon_Mapping) maintains per-pixel photon count, radius and rescaled flux, with a common update parameter of 2/3. It recommends an initial footprint of a few pixels. The current renderer instead averages fresh-map gathers under a globally shrinking world-space radius; do not call a radius tweak canonical SPPM. Multiple diffuse receivers, variable camera allocation and temporal changes need a new accumulation derivation before transplanting per-pixel state.

[Neural Progressive Photon Mapping (Eurographics 2026)](https://justinbenoist.github.io/publications/nppm.html) predicts parameters of an analytical density kernel while retaining progressive bandwidth control. It is an interesting later reconstruction experiment, not a generator of missing light paths. Its [released implementation](https://github.com/JustinBenoist/NPPM) requires CUDA extensions and tiny-cuda-nn, so Metal integration would require a port and training/inference work.

### 3. Manifold sampling offers sharp connections at significant complexity

[Specular Manifold Sampling (2020)](https://rgl.epfl.ch/publications/Zeltner2020Specular) solves constrained specular subpaths between two vertices and supports unbiased and biased variants. It explicitly samples challenging SDS configurations; its showcase still has indirect-caustic noise outside the sampled family. Newton convergence alone does not give the correct weight: multiple solution basins and their sampling probabilities matter. Use the authors' [corrected reference implementation](https://github.com/tizian/specular-manifold-sampling), including the documented angle-periodicity correction.

[Sample Space Partitioning and Spatiotemporal Resampling for SMS (2025)](https://research.nvidia.com/labs/rtr/publication/hong2025partition/) restricts Newton walks near seeds, builds useful priors and reuses solutions. It demonstrates why naive SMS remains costly for interaction. Start with analytic geometry and a small set of transmission/reflection sequences; a faceted teapot adds discrete face changes and does not acquire smooth optical curvature from more samples. Ordinary straight shadow rays through glass cannot replace constrained bent connections.

### 4–5. Reuse and general bidirectional transport are major projects

[ReSTIR BDPT (2025)](https://research.nvidia.com/labs/rtr/publication/hedstrom2025restir/) introduces technique-aware extended path space, hybrid shifts and caustic reservoirs. Its reported approximately 50 ms/frame is an author-system result, not an M1 expectation. [Author code](https://github.com/Shmaug/ReSTIR-BDPT) is a reference for probability accounting, not a Metal library.

[Guided ReSTIR FG+ (EGSR 2026)](https://diglib.eg.org/items/f4bbd38a-fda9-4c43-afc9-dd31e3c5eec1) directly addresses wasted photons in large scenes using GPU histogram guidance and photon resampling. The [official implementation](https://github.com/TU-Clausthal-Rendering/Guided-ReSTIR-FG-Plus) uses Falcor/Windows/DXR, warns that refactored code timings differ from the paper, and excludes photon emission from environment/directional lights. Borrow its guiding ideas before attempting the entire reuse framework.

[VCM (2012), with SmallVCM code and an implementation report](https://iliyan.com/publications/VertexMerging/) combines photon merging and bidirectional connection using MIS. Progressive VCM is consistent and robust across complementary transport families. Plain BDPT is not a universal glass cure: its own showcased reflected caustics motivate merging. The existing narrow partition is easier to audit than replacing it with all VCM strategies. [Practical Path Guiding (2017)](https://jannovak.info/publications/PathGuide/index.html) can improve general incident-direction sampling, but a learned continuous direction distribution does not solve a zero-measure delta-light/specular connection; prioritize emission guiding for the current caustic bottleneck.

## M1 implementation and acceptance criteria

Apple states that hardware ray tracing first reached Macs with [M3](https://www.apple.com/fi/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/). M1 cannot inherit RTX/DXR research timings. GPU-compute histograms, prefix scans and bounded buffers fit the existing WGSL/Metal design; extra traversal, atomics, random memory access and OIDN work still compete for the same GPU. Measure trace, sort, gather, splat contention, guide training and reconstruction separately.

For the next prototype, expose photon budget independently of camera rounds, instrument launched/landed/useful photons and per-pass time, then compare baseline versus prior-round light/caster guidance at equal elapsed time. Preserve the estimator partition and fixed-world/blend scope; stochastic world switching still needs an adjoint transition law. Report raw HDR and displayed error separately for visible floor, through-glass floor and pure-glass regions, across several independent seeds and time budgets. Use a separately converged reference; its remaining noise is an error floor. Include an offscreen caster, a large area emitter, thin adjacent receivers and the analytic index-matched control. Accept guidance only when its training overhead buys lower raw error, rather than merely a smoother denoised picture.
