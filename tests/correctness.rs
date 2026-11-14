use glam::Vec3;
use multiworld_path_tracer::{
    geometry::{aabb, sphere},
    scene,
    settings::{Mode, Settings, View},
};
#[test]
fn sphere_front_inside_tangent_and_miss() {
    let h = sphere(Vec3::new(0., 0., 3.), -Vec3::Z, Vec3::ZERO, 1.).unwrap();
    assert!((h.t - 2.).abs() < 1e-6);
    assert_eq!(h.normal, Vec3::Z);
    let h = sphere(Vec3::ZERO, Vec3::X, Vec3::ZERO, 1.).unwrap();
    assert_eq!(h.t, 1.);
    assert_eq!(h.normal, Vec3::X);
    assert_eq!(
        sphere(Vec3::new(1., 0., 3.), -Vec3::Z, Vec3::ZERO, 1.)
            .unwrap()
            .t,
        3.
    );
    assert!(sphere(Vec3::new(2., 0., 3.), -Vec3::Z, Vec3::ZERO, 1.).is_none());
    assert!(sphere(Vec3::new(0., 0., 3.), Vec3::Z, Vec3::ZERO, 1.).is_none());
}
#[test]
fn sphere_non_unit_direction_and_self_offset() {
    assert_eq!(
        sphere(Vec3::new(0., 0., 3.), -2. * Vec3::Z, Vec3::ZERO, 1.)
            .unwrap()
            .t,
        1.
    );
    assert!(sphere(Vec3::new(0., 1.002, 0.), Vec3::Y, Vec3::ZERO, 1.).is_none());
}
#[test]
fn box_front_inside_parallel_and_boundary() {
    let lo = Vec3::splat(-1.);
    let hi = Vec3::ONE;
    let h = aabb(Vec3::new(0., 0., 3.), -Vec3::Z, lo, hi).unwrap();
    assert_eq!(h.t, 2.);
    assert_eq!(h.normal, Vec3::Z);
    let h = aabb(Vec3::ZERO, -Vec3::X, lo, hi).unwrap();
    assert_eq!(h.t, 1.);
    assert_eq!(h.normal, -Vec3::X);
    assert!(aabb(Vec3::new(2., 0., 3.), -Vec3::Z, lo, hi).is_none());
    assert!(aabb(Vec3::new(0., 0., 3.), Vec3::Z, lo, hi).is_none());
    assert_eq!(aabb(Vec3::new(1., 0., 3.), -Vec3::Z, lo, hi).unwrap().t, 2.);
    assert!(aabb(Vec3::ZERO, Vec3::ZERO, lo, hi).is_none());
}
#[test]
fn scene_is_reproducible_and_independent() {
    let worlds = scene::worlds(42);
    assert_eq!(worlds, scene::worlds(42));
    assert_ne!(worlds, scene::worlds(43));
    for world in &worlds {
        assert_eq!(world.len(), 29);
        assert_eq!(world.iter().filter(|o| o.emission[0] > 0.).count(), 3);
    }
    assert_ne!(worlds[0], worlds[1]);
}
#[test]
fn accumulation_invalidation_distinguishes_display_and_transport() {
    let old = Settings {
        mode: Mode::Blend,
        ..Default::default()
    };
    let mut s = old.clone();
    s.p = 0.8;
    s.exposure = 1.;
    s.spp = 8;
    s.view = View::Normals;
    assert!(!s.invalidates(&old));
    s.yaw += 0.1;
    assert!(s.invalidates(&old));
    let old = Settings::default();
    let mut s = old.clone();
    s.p = 0.8;
    assert!(s.invalidates(&old));
    let mut s = old.clone();
    s.seed += 1;
    assert!(s.invalidates(&old));
    let mut s = old.clone();
    s.light_sampling = false;
    assert!(s.invalidates(&old));
    assert!(!s.history_compatible(&old));
}

#[test]
fn horizontal_orbit_wraps_continuously_in_both_directions() {
    let mut s = Settings {
        yaw: std::f32::consts::PI - 0.01,
        ..Default::default()
    };
    let before = s.camera(1.);
    s.orbit(0.02, 0.);
    assert!((s.yaw - (-std::f32::consts::PI + 0.01)).abs() < 1e-5);
    let after = s.camera(1.);
    assert!(Vec3::from_slice(&before[1][..3]).dot(Vec3::from_slice(&after[1][..3])) > 0.999);
    s.orbit(-0.02, 0.);
    assert!((s.yaw - (std::f32::consts::PI - 0.01)).abs() < 1e-5);
    let original = s.yaw;
    s.set_orbit_yaw(original + std::f32::consts::TAU * 100.);
    assert!((s.yaw - original).abs() < 0.0001);
    for _ in 0..2000 {
        s.orbit(0.1, 0.);
        assert!((-std::f32::consts::PI..std::f32::consts::PI).contains(&s.yaw));
    }
    let last = s.yaw;
    s.set_orbit_yaw(f32::NAN);
    assert_eq!(s.yaw, last);
}

#[test]
fn capsule_body_caps_inside_exit_and_degenerate_axis() {
    use multiworld_path_tracer::geometry::capsule;
    let start = -Vec3::Y;
    let end = Vec3::Y;
    for (origin, direction, t, normal) in [
        (Vec3::Z * 3., -Vec3::Z, 2.5, Vec3::Z),
        (Vec3::Y * 3., -Vec3::Y, 1.5, Vec3::Y),
        (Vec3::ZERO, Vec3::X, 0.5, Vec3::X),
        (Vec3::ZERO, Vec3::Y, 1.5, Vec3::Y),
        (-Vec3::Y * 3., Vec3::Y, 1.5, -Vec3::Y),
    ] {
        let hit = capsule(origin, direction, start, end, 0.5).unwrap();
        assert!((hit.t - t).abs() < 1e-6);
        assert!(hit.normal.abs_diff_eq(normal, 1e-6));
    }
    assert!(capsule(Vec3::Z * 3., Vec3::Z, start, end, 0.5).is_none());
    assert!(capsule(Vec3::ZERO, Vec3::ZERO, start, end, 0.5).is_none());
    let hit = capsule(Vec3::Z * 3., -Vec3::Z, Vec3::ZERO, Vec3::ZERO, 0.5).unwrap();
    assert!((hit.t - 2.5).abs() < 1e-6);
}

#[test]
fn selecting_or_swapping_worlds_invalidates_accumulation_and_history() {
    let old = Settings::default();
    let mut changed = old.clone();
    changed.worlds[1] = scene::WorldPreset::SunkenTemple;
    assert!(changed.invalidates(&old));
    assert!(!changed.history_compatible(&old));
    let mut swapped = old.clone();
    swapped.worlds.swap(0, 1);
    assert!(swapped.invalidates(&old));
    assert!(!swapped.history_compatible(&old));
}

#[test]
fn ellipsoid_flattened_cap_and_inside_exit() {
    use multiworld_path_tracer::geometry::ellipsoid;
    let radii = Vec3::new(2., 0.4, 0.8);
    let h = ellipsoid(Vec3::Y * 3., -Vec3::Y, Vec3::ZERO, radii).unwrap();
    assert!((h.t - 2.6).abs() < 1e-5);
    assert!(h.normal.abs_diff_eq(Vec3::Y, 1e-5));
    let h = ellipsoid(Vec3::ZERO, Vec3::X, Vec3::ZERO, radii).unwrap();
    assert!((h.t - 2.).abs() < 1e-5);
    assert!(h.normal.abs_diff_eq(Vec3::X, 1e-5));
}
