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

/// CPU capsule reference, including exits when the origin is inside the solid.
pub fn capsule(origin: Vec3, direction: Vec3, start: Vec3, end: Vec3, radius: f32) -> Option<Hit> {
    let axis = end - start;
    let length2 = axis.length_squared();
    let oc = origin - start;
    let mut closest = f32::INFINITY;
    let speed2 = direction.length_squared();
    if speed2 == 0. {
        return None;
    }
    if length2 > 1e-12 {
        let along = axis.dot(direction);
        let projected = axis.dot(oc);
        let a = speed2 - along * along / length2;
        let b = oc.dot(direction) - projected * along / length2;
        let c = oc.length_squared() - projected * projected / length2 - radius * radius;
        let disc = b * b - a * c;
        if a > 1e-8 && disc >= 0. {
            for t in [(-b - disc.sqrt()) / a, (-b + disc.sqrt()) / a] {
                let y = projected + t * along;
                if t > EPSILON && y >= 0. && y <= length2 {
                    closest = closest.min(t);
                }
            }
        }
    }
    for (index, center) in [start, end].into_iter().enumerate() {
        let delta = origin - center;
        let b = delta.dot(direction);
        let disc = b * b - speed2 * (delta.length_squared() - radius * radius);
        if disc >= 0. {
            for t in [(-b - disc.sqrt()) / speed2, (-b + disc.sqrt()) / speed2] {
                let y = (origin + t * direction - start).dot(axis);
                if t > EPSILON
                    && (length2 <= 1e-12 || (index == 0 && y <= 0.) || (index == 1 && y >= length2))
                {
                    closest = closest.min(t);
                }
            }
        }
    }
    closest.is_finite().then(|| {
        let point = origin + closest * direction;
        let u = if length2 > 1e-12 {
            ((point - start).dot(axis) / length2).clamp(0., 1.)
        } else {
            0.
        };
        Hit {
            t: closest,
            normal: (point - start - u * axis).normalize(),
        }
    })
}

pub fn ellipsoid(origin: Vec3, direction: Vec3, center: Vec3, radii: Vec3) -> Option<Hit> {
    let local_origin = (origin - center) / radii;
    let local_direction = direction / radii;
    let hit = sphere(local_origin, local_direction, Vec3::ZERO, 1.)?;
    Some(Hit {
        t: hit.t,
        normal: ((origin + hit.t * direction - center) / (radii * radii)).normalize(),
    })
}
