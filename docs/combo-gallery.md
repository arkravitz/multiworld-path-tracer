# World-collision gallery recipes

The README gallery contains six selected compositions from an 18-shot exploration. All six are actual 960×720 GPU renders at 1,024 spp, seed 42 and depth 12, with MIS on and denoising off. Exposure, Reinhard tone mapping and sRGB conversion are the renderer's normal display/export pipeline. The PNGs are copied directly from its exports.

Regenerate the six selected images and a fresh settings manifest:

```sh
cargo run --release --example combo_gallery -- --output output/combo-gallery/final
```

For the 18-shot exploration, add `--explore --width 400 --height 300 --samples 128`. Use `--only <recipe-name>` to render any one recipe. The full catalog lives in [combo_gallery.rs](../examples/combo_gallery.rs).

The shots deliberately show different transport effects: spatial replacement of whole surfaces, independent per-bounce mixing, and A-visible geometry borrowing B's illumination and secondary reflections. Camera framing, exposure and the world-space probability field vary by shot; exact values follow below.

| Gallery title | Export / recipe |
| --- | --- |
| Overgrown sanctuary | [17-forest-portal-close](images/combos/17-forest-portal-close.png) |
| Neon orrery | [15-two-realities](images/combos/15-two-realities.png) |
| Portal garden | [11-portal-garden](images/combos/11-portal-garden.png) |
| Electric ruins | [03-electric-ruins](images/combos/03-electric-ruins.png) |
| Clockwork sanctuary | [18-orbital-ruins](images/combos/18-orbital-ruins.png) |
| Midnight forest | [05-midnight-forest](images/combos/05-midnight-forest.png) |

GPU: AdapterInfo { name: "Apple M1", vendor: 0, device: 0, device_type: IntegratedGpu, device_pci_bus_id: "", driver: "", driver_info: "", backend: Metal, subgroup_min_size: 4, subgroup_max_size: 64, transient_saves_memory: Some(true), limit_bucket: None }

## 17-forest-portal-close

960×720, 1024 spp.

```text
Settings {
    mode: Persistent,
    worlds: [
        SunkenTemple,
        BioluminescentGrove,
    ],
    use_bvh: true,
    light_sampling: true,
    adaptive_bvh: true,
    denoise: false,
    denoise_strength: 0.8,
    p: 0.5,
    persistence: 0.88,
    spp: 8,
    depth: 12,
    exposure: 1.0,
    seed: 42,
    view: Radiance,
    spatial: true,
    primary_a: false,
    threshold: 0.5,
    frequency: 0.5,
    softness: 0.14,
    time: 3.8,
    yaw: 0.2,
    pitch: 0.39,
    distance: 14.5,
}
```

## 15-two-realities

960×720, 1024 spp.

```text
Settings {
    mode: PerBounce,
    worlds: [
        LunarObservatory,
        NeonDistrict,
    ],
    use_bvh: true,
    light_sampling: true,
    adaptive_bvh: true,
    denoise: false,
    denoise_strength: 0.8,
    p: 0.5,
    persistence: 0.88,
    spp: 8,
    depth: 12,
    exposure: 0.75,
    seed: 42,
    view: Radiance,
    spatial: false,
    primary_a: false,
    threshold: 0.5,
    frequency: 0.35,
    softness: 0.14,
    time: 0.0,
    yaw: 0.7,
    pitch: 0.4,
    distance: 16.0,
}
```

## 11-portal-garden

960×720, 1024 spp.

```text
Settings {
    mode: Persistent,
    worlds: [
        NeonDistrict,
        BioluminescentGrove,
    ],
    use_bvh: true,
    light_sampling: true,
    adaptive_bvh: true,
    denoise: false,
    denoise_strength: 0.8,
    p: 0.5,
    persistence: 0.88,
    spp: 8,
    depth: 12,
    exposure: 0.75,
    seed: 42,
    view: Radiance,
    spatial: true,
    primary_a: false,
    threshold: 0.5,
    frequency: 0.55,
    softness: 0.14,
    time: 5.8,
    yaw: 0.62,
    pitch: 0.4,
    distance: 16.0,
}
```

## 03-electric-ruins

960×720, 1024 spp.

```text
Settings {
    mode: PerBounce,
    worlds: [
        SunkenTemple,
        NeonDistrict,
    ],
    use_bvh: true,
    light_sampling: true,
    adaptive_bvh: true,
    denoise: false,
    denoise_strength: 0.8,
    p: 0.16,
    persistence: 0.8,
    spp: 8,
    depth: 12,
    exposure: 0.95,
    seed: 42,
    view: Radiance,
    spatial: false,
    primary_a: true,
    threshold: 0.5,
    frequency: 0.35,
    softness: 0.2,
    time: 0.0,
    yaw: 0.62,
    pitch: 0.4,
    distance: 16.0,
}
```

## 18-orbital-ruins

960×720, 1024 spp.

```text
Settings {
    mode: Persistent,
    worlds: [
        LunarObservatory,
        SunkenTemple,
    ],
    use_bvh: true,
    light_sampling: true,
    adaptive_bvh: true,
    denoise: false,
    denoise_strength: 0.8,
    p: 0.5,
    persistence: 0.88,
    spp: 8,
    depth: 12,
    exposure: 0.75,
    seed: 42,
    view: Radiance,
    spatial: true,
    primary_a: false,
    threshold: 0.5,
    frequency: 0.65,
    softness: 0.14,
    time: 7.0,
    yaw: 0.7,
    pitch: 0.4,
    distance: 16.0,
}
```

## 05-midnight-forest

960×720, 1024 spp.

```text
Settings {
    mode: PerBounce,
    worlds: [
        BioluminescentGrove,
        NeonDistrict,
    ],
    use_bvh: true,
    light_sampling: true,
    adaptive_bvh: true,
    denoise: false,
    denoise_strength: 0.8,
    p: 0.16,
    persistence: 0.8,
    spp: 8,
    depth: 12,
    exposure: 1.2,
    seed: 42,
    view: Radiance,
    spatial: false,
    primary_a: true,
    threshold: 0.5,
    frequency: 0.35,
    softness: 0.2,
    time: 0.0,
    yaw: -0.85,
    pitch: 0.4,
    distance: 16.0,
}
```
