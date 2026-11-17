# Live reconstruction and responsive controls

The completed PNG already looked good, but the live view did not: the presenter mixed new raw paths into an older OIDN reconstruction. It also blended the latest raw image at the selected denoise strength. Both operations could put fresh grain back into the preview while export waited for a current reconstruction.

The live presenter now uses **the latest completed snapshot for both sides of the raw/filtered strength blend**. New raw paths keep accumulating in the background. A completed, current-generation reconstruction replaces the displayed snapshot. The denoise-strength setting and final-export rendering are retained. Before the first reconstruction, the ordinary progressive preview remains visible. Exposure and independent-world blend weight still update immediately; scene/camera/resolution changes reject old reconstruction results. Turn Denoise off to inspect the newest raw accumulation.

## The UI stall

A regression harness sends 24 rapid live glass render requests at 800×600 on Apple M1. Before the scheduling fix, all 24 full glass frames queued ahead of GPU completion and took **12.52 seconds** to drain. First-submit CPU time was about 86 ms. After the fix, the same request burst submitted **one** outstanding job, with about **0.46 seconds** left to drain. These are single-run queue measurements, not a universal click-latency or rendering-throughput claim.

`Renderer::render_live` now polls completion without waiting and permits at most one progressive rendering job in flight. Requests arriving while that job runs leave controls active and defer new rendering; the next accepted job uses the latest settings. Heavy caustic, neural and adaptive paths use one camera sample per interactive job, rather than expanding a large samples/frame setting into many full-frame submissions. The corresponding slider is disabled with an explanation. Headless `render`/`accumulate` retain their exact requested sample budgets.

The application continues polling pending work while paused, so a final OIDN result can still replace the preview. The status line labels its frame cadence **UI FPS**; it is not a hardware rendering-time query. A single expensive GPU job can still delay presentation—bounding backlog cannot make that job cheaper.

## Exports

Interactive Save PNG briefly holds accumulation at its requested snapshot while OIDN finishes, then copies the image asynchronously. A background worker handles readback completion and PNG encoding. Save raw PNG uses the same asynchronous path without waiting for OIDN. UI controls keep running. Changing render settings or resetting while an export is still preparing cancels it instead of silently saving the wrong scene. Once the snapshot has been copied for writing, rendering can continue independently.

Only one UI export is active at a time. Failures are reported in the status area. Explicit command-line/test `save_png` and `save_raw_png` keep their synchronous completion contract. The asynchronous path produces byte-identical filtered/raw PNGs in the regression fixture and does not modify raw accumulation.

## Checks

```sh
cargo test --release --lib oidn_live_display -- --ignored --nocapture
cargo test --release --lib interactive_glass_work -- --ignored --nocapture
cargo test --release --lib asynchronous_image_exports -- --ignored --nocapture
```

The display check injects a completed reconstruction, then adds deliberately noisy new samples and changes the global count. The old presenter fails; the corrected one preserves identical display bytes in World A, World B and independent blend at strengths 0, 0.8 and 1. Exposure remains responsive. Existing OIDN quality, export/raw-preservation and stale-scene checks also pass. Native validation at 800×600 confirmed the clean live snapshot, Pause/Resume, and background Save PNG progress/completion in an isolated rebuilt app instance.

For actual glass FPS improvements after fixing latency, see the [research and ranked performance options](glass-interactive-performance.md). Reduced-resolution interaction with full-resolution settling, reducing fixed-capacity photon work, and denoising cadence/quality profiling remain proposed experiments; they are not silently enabled by this display fix.
