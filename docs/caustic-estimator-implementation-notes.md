# Caustic estimator implementation notes

Research date: 2026-11-15. Scope: `src/shaders/caustics.wgsl` and `src/shaders/trace.wgsl`. Equations below are engineering derivations for this renderer, not quotations or claims of a full implementation of a named paper.

## Current estimator contract

`photon_trace` samples a light with probability `light.emission.w`, a surface point uniformly by area, and an emission direction from a full-support mixture. It traces stochastic Fresnel events in importance transport and stores incident photon power already divided by the number of emitted paths N. Visible first-diffuse endpoints are projected by `photon_splat`; diffuse endpoints reached by longer camera prefixes use `caustic_estimate`. Primary gathering returns zero, avoiding overlap with visible splats. The camera suppresses matching D S+ L emission paths. Preserve these ownership rules.

For one light path, write its **N-free** photon power as

    F(u) = Le * A * abs(nL·omega) / (pLight * qDir(omega))
           * absorption * metal-reflectance factors.

Fresnel branch probabilities cancel Fresnel factors because the branch is sampled using the same physical probability. Null/missed/noncaustic paths have F=0. Do not add the camera-side radiance eta-squared factor to this flux. Existing records contain F/N, so reservoir selection must explicitly undo/remove that normalization before applying the output count normalization.

Gathering remains a finite-radius photon-density estimate, including its existing surface/normal filtering bias. Correct reuse can preserve its expectation; it does not remove kernel bias.

## Reservoir reuse with an identity light-path mapping

[GRIS, Lin et al.](https://research.nvidia.com/publication/2022-07_generalized-resampled-importance-sampling-foundations-restir) establishes a framework for correlated reused samples and domain mappings. The following deliberately uses one common primary sample domain, avoiding reconnection Jacobians entirely.

Represent a light path by a uniform, fixed-dimensional primary random vector u (or a reproducible seed interpreted as its finite discrete analogue). Reserve sufficient random dimensions for the maximum path length; early termination ignores unused dimensions. The forward trace, including its inverse direction PDF and stochastic Fresnel choices, maps u to a photon and power F(u). A deterministic seed hash must be treated as a finite discrete approximation, not asserted to be mathematically independent continuous uniforms.

Choose scalar target h(u)>0 everywhere an eventual contribution may be nonzero. A simple choice is epsilon + luminance(F(u)); epsilon must be positive and finite, including for null paths. A spatial target can be epsilon plus power times a receiver-cell kernel. Never use a zero-support local kernel followed by reuse into unrelated receiver cells without additional support/MIS accounting.

For M fresh u_i, stream weights a_i=h(u_i); maintain S=sum(a_i), selecting candidate i with probability a_i/S. Store selected sample Y, its unnormalized power and endpoint, target h(Y), M, and contribution weight

    W = S / (M * h(Y)).

Then E[W*g(Y)] = integral g(u) du for any evaluation g on this same scene/domain. Proof: condition on all candidates; selection expectation equals (1/M)*sum(g(u_i)). This applies to each pixel's visible-splat contribution and each receiver's photon-kernel contribution.

To merge valid source reservoirs i, choose **sample-independent** confidence weights c_i (fixed M, or a predetermined cap of deterministic counts), evaluate the destination target at each selected sample, and stream

    a_i = c_i * W_i * hDst(Y_i)
    Wout = sum(a_i) / (sum(c_i) * hDst(Yout)).

Conditional selection expectation is sum(c_i*W_i*g(Y_i))/sum(c_i). Correlation between source reservoirs does not change this expectation; it can increase variance. A fresh uniform sample is a valid source with W=1 and c=1. Reset/null conventions must preserve the denominator: a trace miss is a legitimate zero-contribution candidate, not permission to divide by the number of successful photons. Positive targets avoid 0/0 and changing-support ambiguity.

Produce K output reservoirs and store photon power `Wout*F(Yout)/K` in each selected endpoint record; rebuild the normal hash grid and invoke the existing splat with that same power. Every output reservoir estimates the whole photon integral, even when its target favors a cell. Do not subsequently discard selected photons merely because they lie outside the reservoir's favored cell. Adding ordinary photons as another estimator requires an explicit convex combination, not summing two full estimates.

Spatial reuse here means reading neighboring reservoirs with different targets over the **same complete light-path domain**; the identity mapping has Jacobian 1. Temporal reuse means reading a prior-frame reservoir. This is genuine reservoir resampling when the weighted selection/normalization occurs. Keeping old photons or retracing old seeds without it is photon recycling.

Reset reservoirs on scene, emission, material, IOR, transport-depth, guiding-distribution, and world epoch changes unless old seeds are explicitly interpreted in a consistently weighted new-domain estimator. Camera movement alone does not change the light-path domain: reproject/retest current visible splats, never reuse old screen splat values. Keep worlds separate. Use ping-pong input/output buffers; reading concurrently overwritten neighbors invalidates the intended pass. Trace fresh seeds each frame and keep count confidences bounded by a deterministic rule to avoid history dominating innovation. Reuse does not promise the variance of the nominal M independent samples.

## Analytic manifold shooting: direct estimator requirements

[Specular Manifold Sampling, Zeltner et al.](https://rgl.epfl.ch/publications/Zeltner2020Specular) connects fixed endpoints using a Newton constraint solve and distinguishes biased and unbiased variants. A finite Newton solve alone supplies neither a complete set of paths nor their probabilities. Its reference implementation also documents a fix for periodic angular residuals. Use tangent-plane residuals below to avoid that particular wrap problem.

For a fixed area-light point y and diffuse receiver x, choose a specific analytic interface sequence, e.g. entry and exit of one glass sphere. Parameterize initial directions by local coordinates z=(z1,z2). Intersect the analytic surfaces, apply deterministic refraction for the chosen transmission chain, then intersect the outgoing ray with the tangent plane at x. Define

    R(z) = (dot(pEnd(z)-x,t1), dot(pEnd(z)-x,t2)).

Newton uses z <- z - inverse(DR)*R with a bounded line search. Reject misses, wrong primitive/face order, TIR for a transmission branch, singular Jacobians, invalid forward distances, and occlusion. Near a candidate root, choose coordinates whose differential direction solid angle is dz1 dz2; otherwise retain Jomega=abs(dot(omega,cross(domega/dz1,domega/dz2))). Then

    J_Aomega = abs(det(DR)) / Jomega.

The root's receiver irradiance contribution, under forward flux transport, is

    Eroot = Le * abs(nL·omega) * T / (pLight * pArea * pChain * pRoot * J_Aomega).

T is the product of transmission Fresnel factors and Beer attenuation for this deterministically chosen chain; pChain is any sampled chain selection probability. Multiply by receiver albedo/pi for diffuse outgoing radiance. No additional receiver cosine belongs outside J_Aomega, which already maps to receiver area. If all roots are enumerated, sum their terms and omit pRoot. If one root is selected, pRoot is its actual discrete probability, including solver seed basin probabilities. Treating the first successful root as probability one loses energy or changes it unpredictably.

Finite differences of R are implementable in WGSL but must remain on the same branch; central differences with relative step sizes should be checked against analytic derivatives. Near fold caustics det(DR) approaches zero; clamping it changes the estimator. A rigorous full smooth-sphere direct SMS estimator therefore remains blocked on complete root handling or a valid inverse-probability estimator, not merely on writing Newton code.

An exact narrow special case is direct double transmission through a parallel planar slab, with light and receiver on opposite sides and equal external IOR. Let a,b be the external normal distances, t slab thickness, rho endpoint lateral separation, theta the external angle, eta=nOutside/nGlass, and thetaT=asin(eta*sin(theta)). Solve the monotone scalar equation

    rho = (a+b)*tan(theta) + t*tan(thetaT)

by bracketed bisection/Newton. For a receiver plane parallel to the slab,

    rhoPrime = (a+b)/cos(theta)^2 + t*eta*cos(theta)/cos(thetaT)^3
    J_Aomega = rho*rhoPrime/sin(theta).

At the axis use the finite limit J_Aomega=(a+b+t*eta)^2. This chain has a unique direct root under the stated assumptions. Validate finite slab boundaries and all blockers; exclude internal-reflection chains from this particular estimator. This does not establish complete root enumeration for spheres, overlapping dielectrics, or arbitrary mesh glass.

## Recommended scoped sphere feature: manifold-guided emission

A useful, energy-preserving alternative is to use a real bounded Newton sphere solve **only to choose a directional proposal center**. Label this feature “Manifold-guided emission (spheres)”, rather than claiming a full direct SMS estimator.

Conditioned on the already sampled light point y and an independently selected prior-frame diffuse focus x, solve the tangent-plane residual above. A converged root defines unit axis a. Randomly sample a **finite** normalized cone around a and mix it with the existing full-support emitter proposal:

    qCone(omega) = indicator(dot(omega,a)>=cos(thetaMax))
                   / (2*pi*(1-cos(thetaMax)))
    qNew(omega) = (1-beta)*qBase(omega) + beta*qCone(omega).

Use qNew for all sampled directions, including those generated by qBase. If Newton fails, use beta=0 and qBase. Then run the actual scene trace and actual stochastic Fresnel handling; do not assume the solver's chain is visible or guaranteed. The root probability and shooting Jacobian are not needed for normalization because the solver only specifies a conditional normalized proposal. Their values can help choose cone width.

The center may depend deterministically on y and prior history, but it must not depend on an unaccounted current raw direction sample that is then transformed toward a root. Selecting focus j before sampling a direction is valid using the conditional full-support mixture qNew(omega|j); no sum over all focus solvers is needed because every j separately supports the entire integral. If instead choosing between several directional components within one mixture, evaluate the sum of the PDFs of all applicable components.

Keep beta<1, enforce a finite minimum cone solid angle, and use stable `1-cos(thetaMax)` arithmetic. A very narrow cone can oversample one tiny receiver patch while increasing noise elsewhere; choose width from a receiver footprint and test equal-time image error. Store branch/object IDs if validating that both sphere crossings match the intended caster. Finite differences near tangency/TIR may fail often; fall back rather than substituting a fake root.

Changing proposal history changes the u-to-path mapping. For cached reservoirs either freeze the proposal within its epoch or retain enough information to re-evaluate the reservoir estimator under the correct mapping. An easy safe implementation stores/reuses already weighted physical photon samples while scene and their source-domain meaning stay fixed, with h evaluated on those samples; do not silently retrace old seeds under a different direction distribution and retain their old W.

## Combining with existing estimators

Photon guiding and manifold-guided cones are normalized emission proposals; they can be nested into qDir with full mixture evaluation. Reservoir reuse then resamples those weighted light paths. These three operations have distinct responsibilities.

A direct SMS contribution duplicates some paths already handled by photon splats/gathering and camera suppression. Before adding it, either partition path classes using actual chain IDs or use complementary estimator weights on the same supported path class. Globally scaling all photons by 1-alpha while adding SMS for only a single sphere/slab class drops the remaining caustics. The present photon record has endpoint and length, not enough chain identity for that partition.

## Required verification

- Constant-contribution discrete reservoir test: fresh RIS and repeated spatial/temporal merges must retain the known mean, including zeros, capped deterministic confidences, and unequal targets.
- Photon-budget and reservoir-output-count sweeps: total power and image mean remain stable when N, K, or reuse passes change. Count all launched paths, including misses.
- Static target with moving camera: reprojected visible splats agree in mean; no old screen-space ghost energy.
- Overlapping proposal cones: directions from each branch must evaluate the same summed PDF; check normalized-mixture integral numerically.
- Sphere solver: Snell residual, intended entry/exit IDs, reversibility, tangency/TIR failures, and exact fallback behavior. Compare finite-difference Jacobians at several step sizes.
- Slab: match the scalar analytic Jacobian, axial limit, eta=1 straight-line limit, finite-boundary rejection, and Beer/Fresnel energy.
- Equal-time linear-HDR image comparisons against the existing estimator over independent seeds. Report mean/variance separately; lower temporal noise alone can conceal correlation or brightness bias.
