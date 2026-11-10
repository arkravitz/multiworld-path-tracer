//! Run explicitly with: cargo test --test gpu_transport -- --ignored --nocapture
use multiworld_path_tracer::{
    gpu::{self, Pixel, Renderer},
    scene::Object,
    settings::{Mode, Settings},
};
fn render(r: &mut Renderer, s: &Settings, total: u32) -> Vec<Pixel> {
    r.reset();
    let mut settings = s.clone();
    while r.samples < total {
        settings.spp = 32.min(total - r.samples);
        r.render(&settings, true);
    }
    r.read_pixels().unwrap()
}
fn linear(p: &Pixel, mode: Mode, probability: f32, samples: u32) -> [f64; 3] {
    std::array::from_fn(|c| {
        if mode == Mode::Blend {
            (f64::from(p.a[c]) * f64::from(probability)
                + f64::from(p.b[c]) * f64::from(1. - probability))
                / f64::from(samples)
        } else {
            f64::from(p.a[c]) / f64::from(samples)
        }
    })
}
fn mean(p: &[Pixel], mode: Mode, probability: f32, samples: u32) -> f64 {
    p.iter()
        .map(|p| {
            linear(p, mode, probability, samples)
                .into_iter()
                .sum::<f64>()
                / 3.
        })
        .sum::<f64>()
        / p.len() as f64
}
fn assert_same(a: &[Pixel], am: Mode, b: &[Pixel], bm: Mode, p: f32, samples: u32) {
    let max = a
        .iter()
        .zip(b)
        .flat_map(|(a, b)| {
            linear(a, am, p, samples)
                .into_iter()
                .zip(linear(b, bm, p, samples))
                .map(|(a, b)| (a - b).abs())
        })
        .fold(0_f64, f64::max);
    assert!(max < 1e-5, "coupled estimator difference {max}");
}
fn plane(y: f32, color: [f32; 3], emission: [f32; 3]) -> Object {
    Object {
        center_radius: [0., y, 0., 0.],
        extent_shape: [0., 0., 0., 2.],
        color_material: [color[0], color[1], color[2], 0.],
        emission: [emission[0], emission[1], emission[2], 0.],
    }
}
#[test]
#[ignore = "requires a real Metal/Vulkan/DX12 compute adapter"]
fn gpu_transport_endpoints_persistence_and_analytic_hybrid() {
    let (device, queue) = pollster::block_on(gpu::headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 40, 30, 42).unwrap();
    let mut s = Settings::default();
    let budget = 64;
    for (p, fixed) in [(1., Mode::WorldA), (0., Mode::WorldB)] {
        s.p = p;
        s.mode = fixed;
        let reference = render(&mut r, &s, budget);
        for mode in [
            Mode::Blend,
            Mode::PerPath,
            Mode::PerBounce,
            Mode::Persistent,
        ] {
            s.mode = mode;
            let image = render(&mut r, &s, budget);
            assert_same(&reference, fixed, &image, mode, p, budget);
        }
    }
    s.p = 0.5;
    s.mode = Mode::PerPath;
    let path = render(&mut r, &s, budget);
    s.mode = Mode::Persistent;
    s.persistence = 1.;
    let persistent = render(&mut r, &s, budget);
    assert_same(
        &path,
        Mode::PerPath,
        &persistent,
        Mode::Persistent,
        0.5,
        budget,
    );
    s.mode = Mode::PerBounce;
    let bounce = render(&mut r, &s, budget);
    s.mode = Mode::Persistent;
    s.persistence = 0.;
    let persistent = render(&mut r, &s, budget);
    assert_same(
        &bounce,
        Mode::PerBounce,
        &persistent,
        Mode::Persistent,
        0.5,
        budget,
    );
    // Blend weight and exposure must not trigger accumulation invalidation.
    s.mode = Mode::Blend;
    render(&mut r, &s, 32);
    s.p = 0.8;
    s.exposure = 1.;
    r.render(&s, false);
    assert_eq!(r.samples, 32);
    // Partition independence, using the shader's absolute sample index.
    s = Settings::default();
    s.spp = 1;
    let first = render(&mut r, &s, 32);
    r.reset();
    for _ in 0..32 {
        r.render(&s, true);
    }
    let second = r.read_pixels().unwrap();
    assert_same(&first, s.mode, &second, s.mode, s.p, 32);
    // A = diffuse floor; B = luminous ceiling. Every primary ray sees one plane.
    // Once per path: p*a*env + (1-p)*E.
    // Per bounce: p*a*(p*env + (1-p)*E) + (1-p)*E.
    // Cross-world A-floor -> B-ceiling is absent from either independent world.
    r.set_worlds(&[
        vec![plane(0., [0.8; 3], [0.; 3])],
        vec![plane(3., [0.; 3], [2., 1., 0.5])],
    ])
    .unwrap();
    s = Settings {
        depth: 4,
        p: 0.5,
        ..Default::default()
    };
    let budget = 512;
    let mut values = [0.; 3];
    for (i, mode) in [Mode::Blend, Mode::PerPath, Mode::PerBounce]
        .into_iter()
        .enumerate()
    {
        s.mode = mode;
        let image = render(&mut r, &s, budget);
        values[i] = mean(&image, mode, s.p, budget);
        assert!(
            image
                .iter()
                .flat_map(|p| p.a)
                .all(|v| v.is_finite() && v >= 0.)
        );
    }
    let env = (0.045 + 0.055 + 0.075) / 3.;
    let emission = (2. + 1. + 0.5) / 3.;
    let independent = 0.5 * 0.8 * env + 0.5 * emission;
    let hybrid = 0.5 * 0.8 * (0.5 * env + 0.5 * emission) + 0.5 * emission;
    eprintln!(
        "GPU plane means: blend={:.6}, per-path={:.6}, per-bounce={:.6}; analytic={independent:.6}, {hybrid:.6}",
        values[0], values[1], values[2]
    );
    assert!((values[0] - independent).abs() < 0.0001);
    // ~600k samples; conservative absolute tolerance for Monte Carlo variance.
    assert!((values[1] - values[0]).abs() < 0.008);
    assert!((values[2] - hybrid).abs() < 0.008);
    assert!(values[2] - values[0] > 0.18);
    // View-only changes preserve radiance, including while paused.
    let before = r.read_pixels().unwrap();
    for view in multiworld_path_tracer::settings::View::ALL {
        s.view = view;
        r.render(&s, false);
        assert_eq!(r.samples, budget);
    }
    assert_same(
        &before,
        s.mode,
        &r.read_pixels().unwrap(),
        s.mode,
        s.p,
        budget,
    );
    // Seed changes restore the full generated worlds after a custom fixture.
    s.seed = 43;
    s.view = multiworld_path_tracer::settings::View::Radiance;
    s.spp = 16;
    r.render(&s, true);
    let restored = r.read_pixels().unwrap();
    let (device, queue) = (r.device.clone(), r.queue.clone());
    let mut reference = Renderer::new(device, queue, 40, 30, 43).unwrap();
    reference.render(&s, true);
    assert_same(
        &restored,
        s.mode,
        &reference.read_pixels().unwrap(),
        s.mode,
        s.p,
        16,
    );
    // Exercise target recreation, noise, and the depth-dependent primary lock.
    r.resize(32, 24).unwrap();
    s.spatial = true;
    s.primary_a = true;
    r.render(&s, true);
    assert_eq!(r.samples, 16);
    let image = r.read_pixels().unwrap();
    assert_eq!(image.len(), 32 * 24);
    assert!(image.iter().all(|p| (p.stats_a[0] - 16.).abs() < 0.001));
    assert!(
        image
            .iter()
            .flat_map(|p| p.a)
            .all(|v| v.is_finite() && v >= 0.)
    );
}
