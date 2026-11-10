# Gallery selection

The capture run explored 14 outputs: the two isolated worlds, ordinary blend,
once-per-path and per-bounce selection, high-persistence switching, primary-A
lighting, two spatial-noise camera angles, three debug views, and a paired
32-spp raw/filtered view. Every image was inspected before selecting the gallery.

Selected:

- **Spatial orbit (hero):** the clearest separation of sphere and box regions,
  with warm/cool indirect light, metallic highlights and colored ground shadows.
  This also demonstrates the surface-anchored spatial field from another angle.
- **Spheres/cubes:** useful, visually distinct reference worlds. They explain the
  experiment before introducing the superimposed geometry.
- **Blend/independent:** matched-camera comparison of the baseline and the hybrid
  transport process. Monte Carlo noise remains visible; the numerical analytic
  test, rather than the image alone, establishes differing expected radiance.
- **Borrowed light:** the strongest illustration of the research question:
  visible spheres remain fixed while box-world secondary transport changes
  illumination and occlusion. Selected for its cyan lighting contrast.
- **Primary-world and normal views:** readable diagnostics that explain how the
  experiment works, rather than treating all unusual silhouettes as artifacts.
- **Raw/filtered at 32 spp:** an honest same-budget preview comparison. The
  conservative filter reduces fine grain but retains larger noise and fireflies;
  it is labeled as filtered and never used as a transport reference.

Not selected for the README:

- The once-per-path and high-persistence images are visually redundant with the
  blend at this camera and budget, despite being useful convergence experiments.
- The other spatial angle is less clear than the hero's foreground separation.
- The switch-count view has low contrast in these mostly short paths, so it does
  not explain the behavior as clearly as primary-world selection or normals.

The unselected captures stay in ignored `output/gallery/`. The reproducible
capture code remains in `examples/gallery.rs`; exact settings are recorded in
`gallery-settings.md`. No AI artwork, desktop content or personal information
was added to the renderer's captures.
