//! Per-world, deterministic surface-area BVH. Flat escape links make traversal
//! stackless: no fixed GPU stack limit and no state carried across ray segments.
use crate::scene::Object;
use anyhow::{Result, ensure};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

pub const INTERIOR: u32 = u32::MAX;
/// 32 bytes, matching WGSL's vec3 + u32 pairs. `escape` skips this subtree.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Node {
    pub low: [f32; 3],
    pub escape: u32,
    pub high: [f32; 3],
    pub object: u32,
}
#[derive(Debug, PartialEq)]
pub struct Bvh {
    /// Unbounded planes first, followed by the finite tree in depth-first order.
    pub nodes: Vec<Node>,
    pub plane_count: u32,
}
#[derive(Clone, Copy)]
struct Primitive {
    id: u32,
    low: Vec3,
    high: Vec3,
}
impl Primitive {
    fn center(self, axis: usize) -> f32 {
        self.low[axis] * 0.5 + self.high[axis] * 0.5
    }
}
fn area(low: Vec3, high: Vec3) -> f64 {
    let e = (high - low).as_dvec3();
    2.0 * (e.x * e.y + e.x * e.z + e.y * e.z)
}
fn bounds(primitives: &[Primitive]) -> (Vec3, Vec3) {
    primitives.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(lo, hi), p| (lo.min(p.low), hi.max(p.high)),
    )
}
impl Bvh {
    /// Build without reordering the scene. Original indices resolve equal-distance
    /// hits identically to the linear query, even for overlapping geometry.
    pub fn build(objects: &[Object]) -> Result<Self> {
        ensure!(
            objects.len() < (u32::MAX / 2) as usize,
            "World has too many objects"
        );
        let mut nodes = Vec::new();
        let mut primitives = Vec::new();
        for (id, object) in objects.iter().enumerate() {
            ensure!(
                bytemuck::cast_slice::<Object, f32>(std::slice::from_ref(object))
                    .iter()
                    .all(|x| x.is_finite()),
                "Object {id} contains non-finite values"
            );
            let shape = object.extent_shape[3];
            ensure!(
                shape == 0.0 || shape == 1.0 || shape == 2.0,
                "Object {id} has an unsupported shape"
            );
            if shape == 2.0 {
                nodes.push(Node {
                    low: [0.; 3],
                    high: [0.; 3],
                    escape: 0,
                    object: id as u32,
                });
                continue;
            }
            let center = Vec3::from_slice(&object.center_radius[..3]);
            let extent = if shape == 0.0 {
                ensure!(
                    object.center_radius[3] > 0.0,
                    "Sphere {id} must have positive radius"
                );
                Vec3::splat(object.center_radius[3])
            } else {
                let e = Vec3::from_slice(&object.extent_shape[..3]);
                ensure!(
                    e.cmpge(Vec3::ZERO).all(),
                    "Box {id} has negative half extents"
                );
                e
            };
            // Outward padding avoids rejecting grazing primitive hits due to f32
            // bound construction or reciprocal slab rounding on different GPUs.
            let padding = (center.abs() + extent + Vec3::ONE) * (8.0 * f32::EPSILON);
            let low = center - extent - padding;
            let high = center + extent + padding;
            ensure!(
                low.is_finite() && high.is_finite() && (high - low).is_finite(),
                "Object {id} bounds overflow f32"
            );
            primitives.push(Primitive {
                id: id as u32,
                low,
                high,
            });
        }
        let plane_count = nodes.len() as u32;
        if !primitives.is_empty() {
            build_tree(&mut primitives, &mut nodes, 0);
        }
        Ok(Self { nodes, plane_count })
    }
}
fn build_tree(primitives: &mut [Primitive], nodes: &mut Vec<Node>, depth: u32) {
    let (low, high) = bounds(primitives);
    let index = nodes.len();
    nodes.push(Node {
        low: low.to_array(),
        high: high.to_array(),
        escape: 0,
        object: INTERIOR,
    });
    if primitives.len() == 1 {
        nodes[index].object = primitives[0].id;
    } else {
        // Exact sweep SAH is inexpensive for these scenes and deterministic.
        // Bound CPU recursion too: after depth 32, balanced median splits ensure
        // even adversarial custom scenes cannot create linear recursion depth.
        let mut best = (f64::INFINITY, 0, primitives.len() / 2);
        if depth < 32 {
            for axis in 0..3 {
                primitives.sort_unstable_by(|a, b| {
                    a.center(axis)
                        .total_cmp(&b.center(axis))
                        .then(a.id.cmp(&b.id))
                });
                let mut suffix = vec![(Vec3::ZERO, Vec3::ZERO); primitives.len()];
                let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
                for i in (0..primitives.len()).rev() {
                    lo = lo.min(primitives[i].low);
                    hi = hi.max(primitives[i].high);
                    suffix[i] = (lo, hi);
                }
                let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
                for split in 1..primitives.len() {
                    lo = lo.min(primitives[split - 1].low);
                    hi = hi.max(primitives[split - 1].high);
                    let (rlo, rhi) = suffix[split];
                    let cost = area(lo, hi) * split as f64
                        + area(rlo, rhi) * (primitives.len() - split) as f64;
                    if cost < best.0 {
                        best = (cost, axis, split);
                    }
                }
            }
        } else {
            let e = high - low;
            best.1 = if e.x >= e.y && e.x >= e.z {
                0
            } else if e.y >= e.z {
                1
            } else {
                2
            };
        }
        primitives.sort_unstable_by(|a, b| {
            a.center(best.1)
                .total_cmp(&b.center(best.1))
                .then(a.id.cmp(&b.id))
        });
        let (left, right) = primitives.split_at_mut(best.2);
        build_tree(left, nodes, depth + 1);
        build_tree(right, nodes, depth + 1);
    }
    nodes[index].escape = nodes.len() as u32;
}
