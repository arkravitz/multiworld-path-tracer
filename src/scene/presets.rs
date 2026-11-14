//! Compact procedural sets, composed around the same origin for cross-world play.
use super::{Object, Rng};
use glam::Vec3;
use std::f32::consts::{PI, TAU};

fn sphere(p: [f32; 3], radius: f32, color: [f32; 3], metal: bool) -> Object {
    Object {
        center_radius: [p[0], p[1], p[2], radius],
        extent_shape: [0.; 4],
        color_material: [color[0], color[1], color[2], if metal { 1. } else { 0. }],
        emission: [0.; 4],
    }
}
fn block(p: [f32; 3], half: [f32; 3], color: [f32; 3], metal: bool) -> Object {
    Object {
        extent_shape: [half[0], half[1], half[2], 1.],
        ..sphere(p, 0., color, metal)
    }
}
fn ellipsoid(p: [f32; 3], radii: [f32; 3], color: [f32; 3]) -> Object {
    Object {
        extent_shape: [radii[0], radii[1], radii[2], 4.],
        ..sphere(p, 0., color, false)
    }
}
fn beam(a: Vec3, b: Vec3, radius: f32, color: [f32; 3], metal: bool) -> Object {
    let p = (a + b) * 0.5;
    let half = (b - a) * 0.5;
    Object {
        extent_shape: [half.x, half.y, half.z, 3.],
        ..sphere(p.to_array(), radius, color, metal)
    }
}
fn lamp(mut object: Object, emission: [f32; 3]) -> Object {
    object.emission = [emission[0], emission[1], emission[2], 0.];
    object
}
fn floor(color: [f32; 3]) -> Vec<Object> {
    vec![Object {
        extent_shape: [0., 0., 0., 2.],
        ..sphere([0.; 3], 0., color, false)
    }]
}
fn arc(
    objects: &mut Vec<Object>,
    center: Vec3,
    basis: [Vec3; 2],
    radii: [f32; 2],
    color: [f32; 3],
    metal: bool,
    angle: f32,
) {
    let [u, v] = basis;
    let [radius, tube] = radii;
    let steps = (angle * radius / 0.35).ceil() as usize;
    for i in 0..steps {
        let point = |t: f32| center + radius * (u * t.cos() + v * t.sin());
        objects.push(beam(
            point(angle * i as f32 / steps as f32),
            point(angle * (i + 1) as f32 / steps as f32),
            tube,
            color,
            metal,
        ));
    }
}

pub(super) fn temple(seed: u32) -> Vec<Object> {
    let mut rng = Rng(seed ^ 0x73a6b1d5);
    let sand = [0.64, 0.39, 0.20];
    let pale = [0.78, 0.64, 0.43];
    let gold = [0.88, 0.65, 0.27];
    let mut o = floor([0.16, 0.105, 0.07]);
    // Terraced sanctuary and a repeated arcade, open from either side.
    for (y, half) in [
        (0.12, [4.9, 0.12, 4.5]),
        (0.36, [4.2, 0.12, 3.9]),
        (0.60, [3.45, 0.12, 3.25]),
    ] {
        o.push(block([0., y, -0.6], half, pale, false));
    }
    for z in [-2.9, 0.0, 2.9] {
        for x in [-2.65, 2.65] {
            o.push(block([x, 0.92, z], [0.54, 0.22, 0.54], sand, false));
            o.push(beam(
                Vec3::new(x, 1.05, z),
                Vec3::new(x, 2.5, z),
                0.30,
                pale,
                false,
            ));
            o.push(block([x, 2.48, z], [0.48, 0.18, 0.47], sand, false));
            // Deliberate stone bands rather than one featureless shaft.
            for y in [1.28, 1.78, 2.27] {
                o.push(block([x, y, z], [0.33, 0.055, 0.33], sand, false));
            }
        }
        arc(
            &mut o,
            Vec3::new(0., 2.5, z),
            [Vec3::X, Vec3::Y],
            [2.65, 0.27],
            pale,
            false,
            PI,
        );
    }
    // Central altar and suspended golden sun.
    o.push(block([0., 0.99, 0.], [1.05, 0.25, 1.05], sand, false));
    o.push(block([0., 1.27, 0.], [1.18, 0.04, 1.18], gold, true));
    o.push(sphere([0., 2.28, 0.], 0.82, gold, true));
    arc(
        &mut o,
        Vec3::new(0., 2.28, 0.),
        [Vec3::X, Vec3::Y],
        [1.16, 0.055],
        gold,
        true,
        TAU,
    );
    for side in [-1., 1.] {
        o.push(lamp(
            block([side * 3.65, 0.51, 0.], [0.07, 0.035, 3.35], [0.; 3], false),
            [8., 2.4, 0.5],
        ));
        for z in [-3.3, 3.3] {
            o.push(block([side * 3.8, 0.9, z], [0.32, 0.42, 0.32], sand, false));
            o.push(lamp(
                sphere([side * 3.8, 1.43, z], 0.26, [0.; 3], false),
                [11., 4., 0.9],
            ));
        }
    }
    // Fallen stones and buried reflective relics around the platform.
    for _ in 0..18 {
        let a = rng.range(0., TAU);
        let r = rng.range(4.8, 5.8);
        let h = rng.range(0.08, 0.25);
        o.push(block(
            [a.cos() * r, h, a.sin() * r],
            [rng.range(0.14, 0.4), h, rng.range(0.15, 0.4)],
            sand,
            false,
        ));
    }
    o.push(lamp(
        sphere([-3., 10., 1.5], 2., [0.; 3], false),
        [11., 8.5, 5.5],
    ));
    o
}

pub(super) fn neon(seed: u32) -> Vec<Object> {
    let mut rng = Rng(seed ^ 0xa8713c2f);
    let black = [0.07, 0.085, 0.13];
    let cyan = [0.10, 0.62, 0.70];
    let violet = [0.40, 0.09, 0.32];
    let mut o = floor([0.10, 0.13, 0.18]);
    o.push(block([0., 0.14, 0.], [5.1, 0.14, 5.1], black, false));
    // A navigable plaza, with a sculptural stepped skyline around its perimeter.
    for (x, z) in [
        (-3.9, -3.7),
        (-1.35, -4.0),
        (1.4, -3.9),
        (3.9, -3.7),
        (-4.0, -0.9),
        (4.0, -0.9),
        (-4.1, 2.1),
        (4.1, 2.1),
    ] {
        let h = rng.range(1.6, 2.7);
        o.push(block([x, h + 0.28, z], [0.72, h, 0.72], black, false));
        o.push(block(
            [x, h * 2. + 0.42, z],
            [0.84, 0.10, 0.84],
            cyan,
            false,
        ));
        for level in 0..(h * 2.) as usize {
            let y = 0.75 + level as f32 * 0.8;
            for dx in [-0.38, 0.0, 0.38] {
                o.push(block(
                    [x + dx, y, z + 0.735],
                    [0.085, 0.19, 0.018],
                    if level % 2 == 0 { cyan } else { violet },
                    false,
                ));
            }
        }
    }
    // Portal frames and levitating chrome satellites in the open center.
    for (z, w, h) in [(-2.5, 2.3, 3.8), (0., 1.85, 3.2), (2.5, 1.45, 2.6)] {
        for x in [-w, w] {
            o.push(block(
                [x, h * 0.5 + 0.3, z],
                [0.16, h * 0.5, 0.18],
                violet,
                false,
            ));
        }
        o.push(block([0., h + 0.3, z], [w + 0.16, 0.16, 0.18], cyan, false));
    }
    for (p, r) in [
        ([0., 1.8, 0.], 0.9),
        ([-2.1, 0.85, 2.], 0.55),
        ([1.8, 2.4, -1.8], 0.48),
    ] {
        o.push(sphere(p, r, [0.80, 0.88, 0.94], true));
    }
    for (x, color) in [(-2.9, [1.8, 10., 15.]), (2.9, [14., 1., 7.])] {
        o.push(lamp(
            block([x, 0.32, 0.], [0.075, 0.035, 4.75], [0.; 3], false),
            color,
        ));
    }
    o.push(lamp(
        block([-3., 3.6, -3.], [0.06, 1.1, 0.65], [0.; 3], false),
        [1.5, 9., 13.],
    ));
    o.push(lamp(
        block([3., 3.2, 0.], [0.06, 1.25, 0.85], [0.; 3], false),
        [11., 0.8, 5.],
    ));
    o.push(lamp(
        sphere([0., 4.7, -2.6], 0.65, [0.; 3], false),
        [6., 3., 10.],
    ));
    o
}

pub(super) fn observatory(seed: u32) -> Vec<Object> {
    let mut rng = Rng(seed ^ 0x19c437ad);
    let white = [0.72, 0.78, 0.82];
    let navy = [0.075, 0.11, 0.18];
    let silver = [0.88, 0.91, 0.95];
    let mut o = floor([0.10, 0.14, 0.21]);
    // Radial terraces and an orrery with intersecting tilted orbital rings.
    for i in 0..12 {
        let a = TAU * i as f32 / 12.;
        let x = a.cos() * 3.7;
        let z = a.sin() * 3.7;
        o.push(block([x, 0.22, z], [0.67, 0.22, 0.67], white, false));
        o.push(sphere([x, 0.60, z], 0.24, silver, true));
    }
    o.push(block([0., 0.22, 0.], [2.3, 0.22, 2.3], white, false));
    o.push(block([0., 0.58, 0.], [1.35, 0.14, 1.35], navy, false));
    let center = Vec3::new(0., 2.65, 0.);
    for (u, v, r) in [
        (Vec3::X, Vec3::Y, 2.15),
        (
            Vec3::new(0.7, 0., 0.714).normalize(),
            Vec3::new(-0.45, 0.78, 0.44).normalize(),
            1.80,
        ),
        (
            Vec3::new(0.6, 0.6, -0.529).normalize(),
            Vec3::new(0.66, -0.75, -0.10).normalize(),
            2.40,
        ),
    ] {
        arc(&mut o, center, [u, v], [r, 0.065], silver, true, TAU);
    }
    o.push(sphere(center.to_array(), 0.85, [0.94, 0.72, 0.30], true));
    for (p, r, color) in [
        ([-1.75, 3.8, 0.], 0.32, [0.12, 0.38, 0.66]),
        ([1.6, 1.4, 0.8], 0.44, [0.7, 0.21, 0.1]),
        ([0.4, 4.35, -0.5], 0.25, [0.7, 0.75, 0.8]),
    ] {
        o.push(sphere(p, r, color, false));
    }
    // Small pedestal exhibits, avoiding a wall that blocks camera orbit.
    for i in 0..7 {
        let a = TAU * i as f32 / 7. + 0.3;
        let r = 4.7;
        let p = [a.cos() * r, 0.62, a.sin() * r];
        o.push(block(p, [0.40, 0.62, 0.40], white, false));
        o.push(sphere(
            [p[0], 1.5, p[2]],
            rng.range(0.28, 0.39),
            if i % 2 == 0 {
                silver
            } else {
                [0.74, 0.48, 0.2]
            },
            i % 2 == 0,
        ));
    }
    o.push(lamp(
        sphere([-3., 10., 2.], 2., [0.; 3], false),
        [12., 15., 20.],
    ));
    o.push(lamp(
        sphere([4., 3.8, -2.8], 0.75, [0.; 3], false),
        [13., 5.5, 1.8],
    ));
    o.push(lamp(
        block([0., 0.77, 0.], [0.9, 0.035, 0.9], [0.; 3], false),
        [1.2, 4., 9.],
    ));
    o
}

pub(super) fn grove(seed: u32) -> Vec<Object> {
    let mut rng = Rng(seed ^ 0x8b621e43);
    let bark = [0.10, 0.23, 0.24];
    let moss = [0.14, 0.30, 0.20];
    let pale = [0.45, 0.67, 0.56];
    let mut o = floor([0.045, 0.105, 0.10]);
    // Living tree: a twisting trunk, spreading roots and branching lantern limbs.
    let trunk = [
        Vec3::new(0., 0.25, 0.),
        Vec3::new(-0.35, 1.5, 0.),
        Vec3::new(0.25, 2.7, -0.3),
        Vec3::new(0., 3.8, 0.),
    ];
    for (i, pair) in trunk.windows(2).enumerate() {
        o.push(beam(pair[0], pair[1], 0.43 - i as f32 * 0.075, bark, false));
    }
    for i in 0..7 {
        let a = TAU * i as f32 / 7.;
        let dir = Vec3::new(a.cos(), 0., a.sin());
        o.push(beam(
            Vec3::new(0., 0.28, 0.),
            dir * 2.0 + Vec3::Y * 0.13,
            0.16,
            bark,
            false,
        ));
        let start = trunk[2];
        let elbow = dir * 1.6 + Vec3::Y * (3.1 + rng.range(-0.3, 0.6));
        let tip = dir * 2.65 + Vec3::Y * (3.9 + rng.range(-0.3, 0.7));
        o.push(beam(start, elbow, 0.17, bark, false));
        o.push(beam(elbow, tip, 0.095, pale, false));
        // Soft globes give the branches a bioluminescent canopy.
        o.push(sphere(
            tip.to_array(),
            0.48,
            if i % 2 == 0 {
                [0.16, 0.65, 0.56]
            } else {
                [0.48, 0.17, 0.54]
            },
            false,
        ));
    }
    // Distinct mushroom silhouettes: stalk, broad cap and speckled crown.
    for (i, (x, z, h, r)) in [
        (-3.3, -1.7, 1.5, 0.85),
        (2.8, -2.6, 1.9, 1.0),
        (-2.7, 2.6, 1.15, 0.72),
        (2.9, 2.6, 0.9, 0.70),
        (-0.9, -3.5, 0.65, 0.5),
        (0.7, 3.4, 0.5, 0.42),
    ]
    .into_iter()
    .enumerate()
    {
        let base = Vec3::new(x, 0.15, z);
        let tip = Vec3::new(x + 0.18, h, z);
        o.push(beam(base, tip, 0.12, pale, false));
        // Flattened caps overhang narrow stalks, with little pale gill disks.
        o.push(ellipsoid(
            (tip - Vec3::Y * 0.10).to_array(),
            [r * 0.93, r * 0.065, r * 0.93],
            pale,
        ));
        o.push(ellipsoid(
            tip.to_array(),
            [r, r * 0.30, r],
            if i % 2 == 0 {
                [0.13, 0.48, 0.49]
            } else {
                [0.41, 0.13, 0.36]
            },
        ));
        for k in 0..5 {
            let a = TAU * k as f32 / 5.;
            let p = tip + Vec3::new(a.cos() * r * 0.55, r * 0.24, a.sin() * r * 0.55);
            o.push(ellipsoid(
                p.to_array(),
                [0.07, 0.028, 0.07],
                [0.74, 0.8, 0.57],
            ));
        }
        o.push(sphere(
            [tip.x, tip.y + r * 0.30, tip.z],
            0.09,
            [0.8, 0.78, 0.40],
            false,
        ));
    }
    for _ in 0..22 {
        let a = rng.range(0., TAU);
        let r = rng.range(3.7, 5.4);
        let size = rng.range(0.2, 0.48);
        o.push(ellipsoid(
            [r * a.cos(), size * 0.35, r * a.sin()],
            [size, size * 0.55, size * 0.85],
            moss,
        ));
    }
    // Few broad, efficient emitters; decorative detail doesn't create hundreds of lights.
    for (p, r, e) in [
        ([0., 4.45, 0.], 0.65, [1., 9., 7.]),
        ([-3.3, 2., -1.7], 0.34, [3., 5., 12.]),
        ([2.8, 2.5, -2.6], 0.4, [11., 1.8, 6.]),
        ([0., 0.40, 2.1], 0.28, [7., 5., 0.8]),
    ] {
        o.push(lamp(sphere(p, r, [0.; 3], false), e));
    }
    o.push(sphere([1.65, 0.68, 0.6], 0.68, [0.72, 0.91, 0.83], true));
    o.push(lamp(
        sphere([-3., 10., 2.], 2., [0.; 3], false),
        [5., 10., 13.],
    ));
    o
}
