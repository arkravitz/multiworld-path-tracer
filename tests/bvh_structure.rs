use glam::Vec3;
use multiworld_path_tracer::{
    bvh::{Bvh, INTERIOR, Node},
    scene::{self, Object},
};
fn sphere(x: f32) -> Object {
    Object {
        center_radius: [x, 0., 0., 1.],
        extent_shape: [0.; 4],
        color_material: [0.5; 4],
        emission: [0.; 4],
    }
}
fn check(objects: &[Object], bvh: &Bvh) {
    assert_eq!(size_of::<Node>(), 32);
    let mut ids = Vec::new();
    for (i, n) in bvh.nodes.iter().enumerate() {
        if i < (bvh.plane_count as usize) {
            assert_eq!(objects[n.object as usize].extent_shape[3], 2.);
            ids.push(n.object);
            continue;
        }
        assert!(n.escape as usize > i && n.escape as usize <= bvh.nodes.len());
        let low = Vec3::from(n.low);
        let high = Vec3::from(n.high);
        assert!(low.cmple(high).all());
        if n.object == INTERIOR {
            let right = bvh.nodes[i + 1].escape as usize;
            assert!(right < n.escape as usize);
            assert_eq!(bvh.nodes[right].escape, n.escape);
            for child in &bvh.nodes[i + 1..n.escape as usize] {
                assert!(low.cmple(Vec3::from(child.low)).all());
                assert!(high.cmpge(Vec3::from(child.high)).all());
            }
        } else {
            assert_eq!(n.escape as usize, i + 1);
            ids.push(n.object);
            let o = &objects[n.object as usize];
            let c = Vec3::from_slice(&o.center_radius[..3]);
            let e = if o.extent_shape[3] == 0. {
                Vec3::splat(o.center_radius[3])
            } else {
                Vec3::from_slice(&o.extent_shape[..3])
            };
            assert!(low.cmple(c - e).all() && high.cmpge(c + e).all());
        }
    }
    ids.sort_unstable();
    assert_eq!(ids, (0..objects.len() as u32).collect::<Vec<_>>());
    if bvh.nodes.len() > bvh.plane_count as usize {
        assert_eq!(
            bvh.nodes[bvh.plane_count as usize].escape as usize,
            bvh.nodes.len()
        );
    }
}
#[test]
fn deterministic_bvh_contains_every_primitive_and_all_descendants() {
    for seed in [0, 1, 42, 43, u32::MAX] {
        for objects in scene::worlds(seed) {
            let bvh = Bvh::build(&objects).unwrap();
            check(&objects, &bvh);
            assert_eq!(bvh, Bvh::build(&objects).unwrap());
        }
    }
}
#[test]
fn empty_planes_only_singleton_and_coincident_bounds() {
    for objects in [
        vec![],
        vec![sphere(0.)],
        vec![sphere(0.); 129],
        (0..513).map(|i| sphere(i as f32 * 3.)).collect(),
    ] {
        let bvh = Bvh::build(&objects).unwrap();
        check(&objects, &bvh);
    }
    let mut plane = sphere(0.);
    plane.extent_shape[3] = 2.;
    for objects in [
        vec![plane],
        vec![plane, plane],
        vec![sphere(1.), plane, sphere(2.), plane],
    ] {
        let bvh = Bvh::build(&objects).unwrap();
        check(&objects, &bvh);
    }
}
#[test]
fn invalid_geometry_is_rejected_before_upload() {
    let mut bad = sphere(0.);
    bad.center_radius[0] = f32::NAN;
    assert!(Bvh::build(&[bad]).is_err());
    let mut bad = sphere(0.);
    bad.center_radius[3] = -1.;
    assert!(Bvh::build(&[bad]).is_err());
    let mut bad = sphere(0.);
    bad.extent_shape = [-1., 1., 1., 1.];
    assert!(Bvh::build(&[bad]).is_err());
    let mut bad = sphere(0.);
    bad.extent_shape[3] = 3.;
    assert!(Bvh::build(&[bad]).is_err());
    let mut bad = sphere(0.);
    bad.center_radius = [f32::MAX, 0., 0., f32::MAX];
    assert!(Bvh::build(&[bad]).is_err());
}
