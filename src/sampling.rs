//! Upload-time power distribution for finite, two-sided emissive primitives.
//! CDF records occupy the tail of each world's BVH buffer, outside traversal.
use crate::{bvh::Node, scene::Object};

/// Build metadata after scene geometry has passed BVH validation.
pub fn prepare(source: &[Object]) -> (Vec<Object>, Vec<Node>) {
    let mut objects = source.to_vec();
    let weights: Vec<f64> = objects
        .iter()
        .map(|o| {
            let area = match o.extent_shape[3] {
                0.0 => 4.0 * std::f64::consts::PI * f64::from(o.center_radius[3]).powi(2),
                1.0 => {
                    let [x, y, z, _] = o.extent_shape.map(f64::from);
                    8.0 * (x * y + x * z + y * z)
                }
                // Infinite planes have no finite power; cosine paths still sample them.
                _ => 0.0,
            };
            area * (0.2126 * f64::from(o.emission[0])
                + 0.7152 * f64::from(o.emission[1])
                + 0.0722 * f64::from(o.emission[2]))
            .max(0.0)
        })
        .collect();
    let total: f64 = weights.iter().sum();
    let mut records = Vec::new();
    let mut cdf = 0.0_f64;
    let mut previous = 0.0_f32;
    for (index, (object, weight)) in objects.iter_mut().zip(weights).enumerate() {
        object.emission[3] = 0.;
        if weight <= 0. || total <= 0. {
            continue;
        }
        cdf += weight / total;
        let end = cdf as f32;
        // Use the actual representable CDF interval as the selection probability.
        let pmf = end - previous;
        if pmf <= 0. {
            continue;
        }
        object.emission[3] = pmf;
        records.push(Node {
            low: [end, pmf, 0.],
            escape: 0,
            high: [0.; 3],
            object: index as u32,
        });
        previous = end;
    }
    (objects, records)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn power_cdf_is_normalized_and_preserves_geometry() {
        for world in crate::scene::worlds(42) {
            let (objects, lights) = prepare(&world);
            assert_eq!(lights.len(), 3);
            assert_eq!(lights.last().unwrap().low[0], 1.);
            let mut previous = 0.;
            for light in lights {
                assert!(light.low[0] > previous);
                assert_eq!(light.low[1], light.low[0] - previous);
                assert_eq!(objects[light.object as usize].emission[3], light.low[1]);
                previous = light.low[0];
            }
            for (mut actual, expected) in objects.into_iter().zip(world) {
                actual.emission[3] = expected.emission[3];
                assert_eq!(actual, expected);
            }
        }
        // Signed radiance remains a valid cosine-path contribution; it must
        // never produce a negative selection probability for other lights.
        let mut signed = crate::scene::worlds(42)[0].clone();
        signed[1].emission = [-100.; 4];
        let (_, records) = prepare(&signed);
        assert_eq!(records.len(), 3);
        assert_eq!(records.last().unwrap().low[0], 1.);
        assert!(records.iter().all(|r| r.low[1] > 0.));
        assert!(prepare(&[]).1.is_empty());
        let mut planes = crate::scene::worlds(42)[0][..1].to_vec();
        planes[0].emission = [10.; 4];
        let (objects, lights) = prepare(&planes);
        assert!(lights.is_empty());
        assert_eq!(objects[0].emission[3], 0.);
    }
}
