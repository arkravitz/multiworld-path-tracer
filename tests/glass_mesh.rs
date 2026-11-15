use glam::Vec3;
use multiworld_path_tracer::{
    geometry,
    scene::{self, GlassPreset, WorldPreset},
    settings::Settings,
};
#[test]
fn obj_parser_and_triangle_reference_cover_backfaces_edges_and_invalid_indices() {
    let source = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1 # face\n";
    let m = scene::mesh::parse_obj(source).unwrap();
    assert_eq!(m.len(), 1);
    for z in [-1., 1.] {
        let h = geometry::triangle(
            Vec3::new(0.25, 0.25, z),
            Vec3::new(0., 0., -z),
            m[0][0],
            m[0][1],
            m[0][2],
        )
        .unwrap();
        assert!((h.t - 1.).abs() < 1e-6);
        assert_eq!(h.normal, Vec3::Z);
    }
    assert!(
        geometry::triangle(Vec3::new(1., 1., 1.), -Vec3::Z, m[0][0], m[0][1], m[0][2]).is_none()
    );
    assert!(scene::mesh::parse_obj("v 0 0 0\nf 1 2 3").is_err());
}
#[test]
fn bundled_online_meshes_are_watertight_and_non_degenerate() {
    use std::collections::BTreeMap;
    for mesh in [scene::mesh::teapot(), scene::mesh::suzanne()] {
        assert!(mesh.len() > 1000);
        let key = |p: Vec3| p.to_array().map(f32::to_bits);
        let mut edges = BTreeMap::new();
        let mut volume = 0.;
        for [a, b, c] in mesh {
            assert!((*b - *a).cross(*c - *a).length_squared() > 1e-16);
            volume += a.dot(b.cross(*c)) / 6.;
            for (a, b) in [(a, b), (b, c), (c, a)] {
                let a = key(*a);
                let b = key(*b);
                let pair = if a < b { (a, b) } else { (b, a) };
                *edges.entry(pair).or_insert(0) += 1;
            }
        }
        assert!(volume > 0.);
        assert!(edges.values().all(|n| *n == 2));
    }
}
#[test]
fn glass_selection_rebuilds_scene_and_invalidates_learned_lighting() {
    let old = Settings {
        worlds: [WorldPreset::GlassGallery, WorldPreset::CrystalGarden],
        ..Default::default()
    };
    let mut new = old.clone();
    new.glass[0] = GlassPreset::Emerald;
    assert!(new.invalidates(&old));
    assert!(!new.history_compatible(&old));
    let worlds = scene::selected_worlds_with_glass(42, new.worlds, new.glass);
    assert!(
        worlds[0]
            .iter()
            .filter(|o| o.color_material[3] >= 2.)
            .all(|o| o.color_material == GlassPreset::Emerald.material())
    );
}

#[test]
fn invalid_dielectric_and_overflowing_triangle_are_rejected() {
    use multiworld_path_tracer::{bvh::Bvh, scene::Object};
    let ball = Object {
        center_radius: [0., 0., 0., 1.],
        extent_shape: [0.; 4],
        color_material: [1., 1., 1., 3.5],
        emission: [0.; 4],
    };
    let mut bad = ball;
    bad.color_material[3] = 2.;
    assert!(Bvh::build(&[bad]).is_err());
    let mut bad = ball;
    bad.color_material[0] = 1.5;
    assert!(Bvh::build(&[bad]).is_err());
    let bad = Object {
        center_radius: [f32::MAX, 0., 0., 0.],
        extent_shape: [f32::MAX, 0., 0., 5.],
        color_material: [0.5, 0.5, 0.5, 0.],
        emission: [0., f32::MAX, 0., 0.],
    };
    assert!(Bvh::build(&[bad]).is_err());
}
