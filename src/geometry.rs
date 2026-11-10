//! CPU references for the WGSL intersection contract; the renderer runs on the GPU.
use glam::Vec3;
pub const EPSILON: f32 = 0.001;
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub t: f32,
    pub normal: Vec3,
}
pub fn sphere(origin: Vec3, direction: Vec3, center: Vec3, radius: f32) -> Option<Hit> {
    let oc = origin - center;
    let a = direction.length_squared();
    let b = oc.dot(direction);
    let discriminant = b * b - a * (oc.length_squared() - radius * radius);
    if discriminant < 0. || a == 0. {
        return None;
    }
    let mut t = (-b - discriminant.sqrt()) / a;
    if t <= EPSILON {
        t = (-b + discriminant.sqrt()) / a;
    }
    (t > EPSILON).then(|| Hit {
        t,
        normal: (origin + t * direction - center) / radius,
    })
}
pub fn aabb(origin: Vec3, direction: Vec3, low: Vec3, high: Vec3) -> Option<Hit> {
    let mut near = -f32::INFINITY;
    let mut far = f32::INFINITY;
    let mut near_n = Vec3::ZERO;
    let mut far_n = Vec3::ZERO;
    for axis in 0..3 {
        if direction[axis].abs() < 1e-8 {
            if origin[axis] < low[axis] || origin[axis] > high[axis] {
                return None;
            }
        } else {
            let mut a = (low[axis] - origin[axis]) / direction[axis];
            let mut b = (high[axis] - origin[axis]) / direction[axis];
            let mut n = Vec3::ZERO;
            n[axis] = -1.;
            if a > b {
                std::mem::swap(&mut a, &mut b);
                n = -n;
            }
            if a > near {
                near = a;
                near_n = n;
            }
            if b < far {
                far = b;
                far_n = -n;
            }
            if near > far {
                return None;
            }
        }
    }
    if near > EPSILON {
        Some(Hit {
            t: near,
            normal: near_n,
        })
    } else if far > EPSILON && far.is_finite() {
        Some(Hit {
            t: far,
            normal: far_n,
        })
    } else {
        None
    }
}
