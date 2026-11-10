use bytemuck::{Pod, Zeroable};
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct Object {
    pub center_radius: [f32; 4],
    /// Half extents and shape: 0 sphere, 1 box, 2 horizontal plane.
    pub extent_shape: [f32; 4],
    /// RGB reflectance and material: 0 Lambertian, 1 perfect mirror.
    pub color_material: [f32; 4],
    pub emission: [f32; 4],
}
/// Specified integer PRNG; scene generation is stable across Rust versions.
pub struct Rng(pub u32);
impl Rng {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9e3779b9);
        let mut z = self.0;
        z = (z ^ (z >> 16)).wrapping_mul(0x21f0aaad);
        z = (z ^ (z >> 15)).wrapping_mul(0x735a2d97);
        z ^ (z >> 15)
    }
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16777216.
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.unit()
    }
}
pub fn worlds(seed: u32) -> [Vec<Object>; 2] {
    std::array::from_fn(|world| {
        let mut rng = Rng(seed ^ (world as u32).wrapping_mul(0x85ebca6b));
        let mut objects = vec![Object {
            center_radius: [0.; 4],
            extent_shape: [0., 0., 0., 2.],
            color_material: if world == 0 {
                [0.56, 0.60, 0.64, 0.]
            } else {
                [0.55, 0.49, 0.43, 0.]
            },
            emission: [0.; 4],
        }];
        for z in 0..5 {
            for x in 0..5 {
                let r = rng.range(0.38, 0.82);
                let h = rng.range(0.45, 1.35);
                let px = (x as f32 - 2.) * 2.15 + rng.range(-0.38, 0.38);
                let pz = (z as f32 - 2.) * 2.15 + rng.range(-0.38, 0.38);
                let color = if world == 0 {
                    [
                        rng.range(0.5, 0.9),
                        rng.range(0.12, 0.55),
                        rng.range(0.06, 0.28),
                    ]
                } else {
                    [
                        rng.range(0.06, 0.25),
                        rng.range(0.35, 0.8),
                        rng.range(0.5, 0.9),
                    ]
                };
                objects.push(Object {
                    center_radius: [px, if world == 0 { r } else { h }, pz, r],
                    extent_shape: [r, h, r, world as f32],
                    color_material: [
                        color[0],
                        color[1],
                        color[2],
                        if (x + z * 5) % 5 == 0 { 1. } else { 0. },
                    ],
                    emission: [0.; 4],
                });
            }
        }
        // Broad, low-hanging lights make unassisted indirect sampling practical.
        let positions = [[-4., 4.2, -2.], [3., 4.8, 0.], [0., 3.8, -4.]];
        for (i, pos) in positions.into_iter().enumerate() {
            let rgb = if world == 0 {
                [[13., 5., 2.], [9., 8., 4.], [12., 2., 1.]][i]
            } else {
                [[2., 5., 14.], [2., 12., 10.], [6., 3., 14.]][i]
            };
            objects.push(Object {
                center_radius: [
                    pos[0] + if world == 0 { 0. } else { 0.6 },
                    pos[1],
                    pos[2],
                    0.85,
                ],
                extent_shape: [0.85, 0.3, 0.85, world as f32],
                color_material: [0.; 4],
                emission: [rgb[0], rgb[1], rgb[2], 0.],
            });
        }
        objects
    })
}
