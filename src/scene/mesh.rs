//! Bundled, attributed online meshes. Each flat-shaded triangle uses the existing
//! 64-byte object ABI: origin, edge1, material, edge2 (not emitted radiance).
use super::Object;
use glam::{Mat4, Vec3};
use std::sync::OnceLock;

pub fn parse_obj(source: &str) -> Result<Vec<[Vec3; 3]>, String> {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    for (line_number, line) in source.lines().enumerate() {
        let mut words = line.split('#').next().unwrap_or("").split_whitespace();
        match words.next() {
            Some("v") => {
                let xyz: Vec<f32> = words
                    .take(3)
                    .map(|s| s.parse::<f32>())
                    .collect::<Result<_, _>>()
                    .map_err(|_| format!("Invalid vertex at {}", line_number + 1))?;
                if xyz.len() != 3 || xyz.iter().any(|v| !v.is_finite()) {
                    return Err(format!("Invalid vertex at {}", line_number + 1));
                }
                vertices.push(Vec3::new(xyz[0], xyz[1], xyz[2]));
            }
            Some("f") => {
                let indices: Vec<usize> = words
                    .map(|s| {
                        let value = s
                            .split('/')
                            .next()
                            .unwrap_or("")
                            .parse::<isize>()
                            .map_err(|_| "Invalid face index".to_string())?;
                        let index = if value < 0 {
                            vertices.len() as isize + value
                        } else {
                            value - 1
                        };
                        if value == 0 || index < 0 || index as usize >= vertices.len() {
                            return Err("Face index out of bounds".to_string());
                        }
                        Ok(index as usize)
                    })
                    .collect::<Result<_, _>>()?;
                if indices.len() < 3 {
                    return Err("Face needs three vertices".to_string());
                }
                for pair in indices[1..].windows(2) {
                    let triangle = [vertices[indices[0]], vertices[pair[0]], vertices[pair[1]]];
                    if (triangle[1] - triangle[0])
                        .cross(triangle[2] - triangle[0])
                        .length_squared()
                        > 1e-16
                    {
                        triangles.push(triangle);
                    }
                }
            }
            _ => {}
        }
    }
    if triangles.is_empty() {
        return Err("Mesh contains no triangles".to_string());
    }
    Ok(triangles)
}
pub fn teapot() -> &'static [[Vec3; 3]] {
    static MESH: OnceLock<Vec<[Vec3; 3]>> = OnceLock::new();
    MESH.get_or_init(|| {
        parse_obj(include_str!("../../assets/meshes/utah-teapot-2026.obj"))
            .expect("bundled Utah Teapot must be valid")
    })
}
pub fn suzanne() -> &'static [[Vec3; 3]] {
    static MESH: OnceLock<Vec<[Vec3; 3]>> = OnceLock::new();
    MESH.get_or_init(|| {
        parse_obj(include_str!("../../assets/meshes/suzanne.obj"))
            .expect("bundled Suzanne must be valid")
    })
}
pub fn instance(mesh: &[[Vec3; 3]], transform: Mat4, material: [f32; 4]) -> Vec<Object> {
    mesh.iter()
        .map(|t| {
            let [a, b, c] = t.map(|p| transform.transform_point3(p));
            Object {
                center_radius: a.extend(0.).to_array(),
                extent_shape: (b - a).extend(5.).to_array(),
                color_material: material,
                emission: (c - a).extend(0.).to_array(),
            }
        })
        .collect()
}
