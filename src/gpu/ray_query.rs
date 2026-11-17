//! Optional native ray queries. World-local primitive IDs share the existing
//! BVH allocation, after its logical nodes and light records.
use crate::{bvh::Node, scene::Object};
use bytemuck::Zeroable;
use wgpu::util::DeviceExt;

pub(super) fn supported(device: &wgpu::Device) -> bool {
    device
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
}

pub(super) fn shader(device: &wgpu::Device, source: String) -> String {
    if supported(device) {
        let source = source.replace("fn intersect_hardware(world:u32,ro:vec3<f32>,rd:vec3<f32>,glass_only:bool)->Hit {return intersect_bvh(world,ro,rd);}", include_str!("../shaders/ray_query.wgsl"));
        format!("enable wgpu_ray_query;\n{source}")
    } else {
        // Unsupported devices never parse the experimental extension.
        source
    }
}

pub(super) struct World {
    pub tlas: wgpu::Tlas,
}

impl World {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        objects: &[Object],
        nodes: &mut Vec<Node>,
        finite_nodes: usize,
    ) -> anyhow::Result<Self> {
        let mut triangles = Vec::<[f32; 3]>::new();
        let mut boxes = Vec::<[f32; 6]>::new();
        let mut triangle_ids = Vec::new();
        let mut box_ids = Vec::new();
        for node in &nodes[..finite_nodes] {
            if node.object == u32::MAX {
                continue;
            }
            let object = &objects[node.object as usize];
            if object.extent_shape[3] == 2. {
                continue;
            }
            if object.extent_shape[3] == 5. {
                let a = glam::Vec3::from_slice(&object.center_radius);
                let b = a + glam::Vec3::from_slice(&object.extent_shape);
                let c = a + glam::Vec3::from_slice(&object.emission);
                let center = (a + b + c) / 3.;
                // A thin conservative prism accounts for vertex reconstruction
                // roundoff, shared edges and nearly coplanar reference hits.
                // Metal tests these triangles; the original analytic routine
                // accepts the surface and preserves the renderer's exact ties.
                let shortest = (a - b).length().min((b - c).length()).min((c - a).length());
                let magnitude = a.abs().max(b.abs()).max(c.abs()).max_element().max(1.);
                let padding = 32. * f32::EPSILON * magnitude;
                let scale = 1. + (padding / shortest.max(1e-12)).max(1e-4);
                let [a, b, c] = [a, b, c].map(|v| center + (v - center) * scale);
                let n = (b - a).cross(c - a).normalize() * padding;
                let v = [a - n, b - n, c - n, a + n, b + n, c + n];
                for [i, j, k] in [
                    [0, 2, 1],
                    [3, 4, 5],
                    [0, 1, 4],
                    [0, 4, 3],
                    [1, 2, 5],
                    [1, 5, 4],
                    [2, 0, 3],
                    [2, 3, 5],
                ] {
                    triangles.extend([v[i], v[j], v[k]].map(|v| v.to_array()));
                }
                triangle_ids.push(node.object);
            } else {
                boxes.push([
                    node.low[0],
                    node.low[1],
                    node.low[2],
                    node.high[0],
                    node.high[1],
                    node.high[2],
                ]);
                box_ids.push(node.object);
            }
        }
        let limits = device.limits();
        anyhow::ensure!(
            std::mem::size_of_val(triangles.as_slice()) as u64 <= limits.max_buffer_size
                && std::mem::size_of_val(boxes.as_slice()) as u64 <= limits.max_buffer_size
                && triangles.len() / 3 <= limits.max_blas_primitive_count as usize
                && boxes.len() <= limits.max_blas_primitive_count as usize,
            "World exceeds hardware ray-query geometry limits"
        );
        // Metadata offsets are relative to the end of nodes + light records.
        let box_offset = triangle_ids.len() as u32;
        for id in triangle_ids.into_iter().chain(box_ids) {
            nodes.push(Node {
                object: id,
                ..Node::zeroed()
            });
        }
        anyhow::ensure!(
            std::mem::size_of_val(nodes.as_slice()) as u64
                <= limits.max_storage_buffer_binding_size,
            "World exceeds ray-query metadata limits"
        );
        let flags = wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE;
        let update_mode = wgpu::AccelerationStructureUpdateMode::Build;
        let descriptor = wgpu::CreateBlasDescriptor {
            label: Some("World ray-query geometry"),
            flags,
            update_mode,
        };
        let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("World ray queries"),
            max_instances: 2,
            flags,
            update_mode,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        let identity = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.];
        let mut instance = 0;
        let mut build_buffers = Vec::new();
        if !triangles.is_empty() {
            let size = wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count: triangles.len() as u32,
                index_format: None,
                index_count: None,
                flags: wgpu::AccelerationStructureGeometryFlags::empty(),
            };
            let blas = device.create_blas(
                &descriptor,
                wgpu::BlasGeometrySizeDescriptors::Triangles {
                    descriptors: vec![size.clone()],
                },
            );
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Ray-query vertices"),
                contents: bytemuck::cast_slice(&triangles),
                usage: wgpu::BufferUsages::BLAS_INPUT,
            });
            let entry = wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: &size,
                        vertex_buffer: &buffer,
                        first_vertex: 0,
                        vertex_stride: 12,
                        index_buffer: None,
                        first_index: None,
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            };
            encoder.build_acceleration_structures([&entry], []);
            build_buffers.push(buffer);
            tlas[instance] = Some(wgpu::TlasInstance::new(&blas, identity, 0, 255));
            instance += 1;
        }
        if !boxes.is_empty() {
            let size = wgpu::BlasAABBGeometrySizeDescriptor {
                primitive_count: boxes.len() as u32,
                flags: wgpu::AccelerationStructureGeometryFlags::empty(),
            };
            let blas = device.create_blas(
                &descriptor,
                wgpu::BlasGeometrySizeDescriptors::AABBs {
                    descriptors: vec![size.clone()],
                },
            );
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Ray-query analytic bounds"),
                contents: bytemuck::cast_slice(&boxes),
                usage: wgpu::BufferUsages::BLAS_INPUT,
            });
            let entry = wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::AabbGeometries(vec![wgpu::BlasAabbGeometry {
                    size: &size,
                    stride: 24,
                    aabb_buffer: &buffer,
                    primitive_offset: 0,
                }]),
            };
            encoder.build_acceleration_structures([&entry], []);
            build_buffers.push(buffer);
            tlas[instance] = Some(wgpu::TlasInstance::new(&blas, identity, box_offset, 255));
        }
        // wgpu 30's Metal acceleration-structure barrier is a no-op. A BLAS
        // build and its dependent TLAS build in one encoder can race. Complete
        // this one-time upload before constructing the TLAS; never wait per ray
        // or per rendering frame. Keep build inputs alive until completion.
        let built = queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::Wait {
            submission_index: Some(built),
            timeout: Some(std::time::Duration::from_secs(60)),
        })?;
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.build_acceleration_structures([], [&tlas]);
        queue.submit([encoder.finish()]);
        Ok(Self { tlas })
    }
}
