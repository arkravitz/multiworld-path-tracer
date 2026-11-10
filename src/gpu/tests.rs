use super::*;
use crate::{
    scene::{Object, Rng},
    settings::{Mode, View},
};
use glam::Vec3;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Probe {
    origin: [f32; 4],
    direction: [f32; 4],
    brute_t_normal: [f32; 4],
    brute_color: [f32; 4],
    brute_emission: [f32; 4],
    bvh_t_normal: [f32; 4],
    bvh_color: [f32; 4],
    bvh_emission: [f32; 4],
}
const PROBE_SHADER: &str = r#"
struct Probe {
    origin:vec4<f32>, direction:vec4<f32>,
    brute_t_normal:vec4<f32>, brute_color:vec4<f32>, brute_emission:vec4<f32>,
    bvh_t_normal:vec4<f32>, bvh_color:vec4<f32>, bvh_emission:vec4<f32>,
}
@group(1) @binding(0) var<storage,read_write> probes:array<Probe>;
@compute @workgroup_size(64)
fn probe_queries(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&probes) {return;}
    let p=probes[id.x];let world=u32(p.origin.w);
    let a=intersect_brute(world,p.origin.xyz,p.direction.xyz);
    let b=intersect(world,p.origin.xyz,p.direction.xyz);
    probes[id.x].brute_t_normal=vec4<f32>(a.t,a.normal);
    probes[id.x].brute_color=vec4<f32>(a.color,a.metal);
    probes[id.x].brute_emission=vec4<f32>(a.emission,0.0);
    probes[id.x].bvh_t_normal=vec4<f32>(b.t,b.normal);
    probes[id.x].bvh_color=vec4<f32>(b.color,b.metal);
    probes[id.x].bvh_emission=vec4<f32>(b.emission,0.0);
}
@compute @workgroup_size(64)
fn probe_primary_world(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&probes) {return;}
    let p=probes[id.x];
    let result=trace(p.origin.xyz,p.direction.xyz,u32(p.direction.w),params.transport.x);
    probes[id.x].bvh_t_normal=vec4<f32>(result.stats,0.0);
}
"#;
fn probe(origin: Vec3, direction: Vec3, world: u32) -> Probe {
    Probe {
        origin: origin.extend(world as f32).to_array(),
        direction: direction.extend(0.).to_array(),
        ..Probe::zeroed()
    }
}
fn queries_with_policy(r: &Renderer, probes: &[Probe], policy: u32) -> Vec<Probe> {
    let mut params = Params::zeroed();
    params.controls[3] = policy as f32;
    params.flags = [0, 0, r.worlds[0].object_count, r.worlds[1].object_count];
    params.acceleration = [
        r.worlds[0].node_count,
        r.worlds[1].node_count,
        r.worlds[0].plane_count,
        r.worlds[1].plane_count,
    ];
    dispatch_probes(r, probes, &params, "probe_queries")
}
fn dispatch_probes(r: &Renderer, probes: &[Probe], params: &Params, entry: &str) -> Vec<Probe> {
    r.queue
        .write_buffer(&r.uniform, 0, bytemuck::bytes_of(params));
    let buffer = r
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BVH query probes"),
            contents: bytemuck::cast_slice(probes),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let query_layout = r
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[buffer_entry(
                0,
                wgpu::BufferBindingType::Storage { read_only: false },
            )],
        });
    let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &query_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    let layout = r
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&r.layout), Some(&query_layout)],
            immediate_size: 0,
        });
    let shader = r.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Actual integrator query equivalence"),
        source: wgpu::ShaderSource::Wgsl(
            format!("{}\n{PROBE_SHADER}", include_str!("../shaders/trace.wgsl")).into(),
        ),
    });
    let pipeline = r
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
    let staging = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = r.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &r.bind_group, &[]);
        pass.set_bind_group(1, &bind, &[]);
        pass.dispatch_workgroups((probes.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&buffer, 0, &staging, 0, buffer.size());
    r.read_buffer(encoder, &staging)
        .unwrap()
        .as_chunks::<{ size_of::<Probe>() }>()
        .0
        .iter()
        .map(|p| bytemuck::pod_read_unaligned(p))
        .collect()
}
fn queries(r: &Renderer, probes: &[Probe]) -> Vec<Probe> {
    queries_with_policy(r, probes, 7)
}
fn assert_queries(probes: &[Probe]) {
    for (i, p) in probes.iter().enumerate() {
        let a = [p.brute_t_normal, p.brute_color, p.brute_emission];
        let b = [p.bvh_t_normal, p.bvh_color, p.bvh_emission];
        // Same primitive routines and stable ties: queries should agree exactly.
        assert_eq!(
            a, b,
            "ray {i}, origin {:?}, direction {:?}",
            p.origin, p.direction
        );
        assert!(a.iter().flatten().all(|x| x.is_finite()));
    }
}
fn sample_rays(objects: &[Vec<Object>; 2]) -> Vec<Probe> {
    let mut rays = Vec::new();
    let mut rng = Rng(0x12345678);
    for world in 0..2 {
        for _ in 0..12000 {
            let origin = Vec3::new(
                rng.unit() * 24. - 12.,
                rng.unit() * 14. - 4.,
                rng.unit() * 24. - 12.,
            );
            let direction = Vec3::new(
                rng.unit() * 2. - 1.,
                rng.unit() * 2. - 1.,
                rng.unit() * 2. - 1.,
            )
            .normalize();
            rays.push(probe(origin, direction, world));
        }
        // Axis-parallel, inside-origin, grazing, and near-parallel rays. Origins
        // drawn from BOTH worlds simulate switching into the other world's solid.
        for object in objects.iter().flatten() {
            let center = Vec3::from_slice(&object.center_radius[..3]);
            for axis in 0..3 {
                let mut direction = Vec3::ZERO;
                direction[axis] = 1.;
                rays.push(probe(center, direction, world));
                rays.push(probe(center, -direction, world));
                let mut extent = Vec3::from_slice(&object.extent_shape[..3]);
                if object.extent_shape[3] == 0. {
                    extent = Vec3::splat(object.center_radius[3]);
                }
                let mut origin = center;
                origin[(axis + 1) % 3] += extent[(axis + 1) % 3];
                rays.push(probe(origin - direction * 4., direction, world));
                let mut tiny = direction;
                tiny[(axis + 1) % 3] = 1e-9;
                rays.push(probe(origin - direction * 4., tiny.normalize(), world));
            }
        }
        // Signed zero directions should not lead to NaNs at slab boundaries.
        rays.push(probe(Vec3::ZERO, Vec3::new(-0.0, 0., 1.), world));
    }
    rays
}
fn object(center: [f32; 3], shape: f32, color: [f32; 3]) -> Object {
    Object {
        center_radius: [center[0], center[1], center[2], 1.],
        extent_shape: [1., 1., 1., shape],
        color_material: [color[0], color[1], color[2], 0.],
        emission: [0.; 4],
    }
}
#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn bvh_gpu_queries_match_linear_for_both_worlds_and_edge_cases() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 16, 12, 42).unwrap();
    let mut total = 0;
    let mut fixtures = vec![
        scene::worlds(42),
        scene::worlds(12345),
        [vec![], vec![]],
        [vec![object([0.; 3], 2., [0.5; 3])], vec![]],
    ];
    // Original index wins even when the plane is tested before coincident solids.
    fixtures.push([
        vec![
            object([0.; 3], 0., [1., 0., 0.]),
            object([0.; 3], 1., [0., 1., 0.]),
            object([0., 0., 1.], 2., [0., 0., 1.]),
        ],
        vec![
            object([0.; 3], 1., [1., 0., 0.]),
            object([0.; 3], 0., [0., 1., 0.]),
        ],
    ]);
    // More geometry and unequal counts exercise dynamically sized per-world buffers.
    fixtures.push(std::array::from_fn(|world| {
        (0..(137 + world * 19))
            .map(|i| {
                object(
                    [
                        (i % 13) as f32 * 2. - 12.,
                        (i / 13) as f32 * 2.,
                        world as f32,
                    ],
                    (i % 2) as f32,
                    [i as f32 / 160., 0.5, 0.8],
                )
            })
            .collect()
    }));
    let mut thin = object([0., 0., 1_000_000.], 1., [0.5; 3]);
    thin.extent_shape[0] = 1e-10;
    fixtures.push([vec![thin], vec![]]);
    for fixture in fixtures {
        r.set_worlds(&fixture).unwrap();
        let mut rays = sample_rays(&fixture);
        rays.push(probe(Vec3::ZERO, Vec3::new(1e-9, 0., 1.), 0));
        assert_queries(&queries(&r, &rays));
        assert_queries(&queries_with_policy(&r, &rays, 4));
        total += rays.len() * 2;
    }
    // A rejected second world must leave BOTH previous worlds/resources intact.
    let rays = sample_rays(&scene::worlds(42));
    let before = queries(&r, &rays[..100]);
    let mut invalid = object([0.; 3], 0., [0.5; 3]);
    invalid.center_radius[3] = -1.;
    assert!(r.set_worlds(&[vec![], vec![invalid]]).is_err());
    let after = queries(&r, &rays[..100]);
    assert_eq!(
        bytemuck::cast_slice::<Probe, u8>(&before),
        bytemuck::cast_slice::<Probe, u8>(&after)
    );
    eprintln!("Compared {total} GPU rays, including hit distance, normal, material and emission");
}
fn assert_pixels(a: &[Pixel], b: &[Pixel], context: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        let a = bytemuck::cast_slice::<Pixel, f32>(std::slice::from_ref(a));
        let b = bytemuck::cast_slice::<Pixel, f32>(std::slice::from_ref(b));
        for (a, b) in a.iter().zip(b) {
            assert!(a.is_finite() && b.is_finite());
            assert!((a - b).abs() <= 1e-6, "{context}, pixel {i}: {a} != {b}");
        }
    }
}
#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn bvh_gpu_transport_and_deferred_display_preserve_all_accumulators() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 48, 32, 42).unwrap();
    for seed in [42, 43] {
        for mode in Mode::ALL {
            for (p, s, spatial, primary) in [
                (0., 0., false, false),
                (1., 1., false, false),
                (0.37, 0.8, false, false),
                (0.5, 0., true, true),
            ] {
                let mut settings = Settings {
                    seed,
                    mode,
                    p,
                    persistence: s,
                    spatial,
                    primary_a: primary,
                    spp: 32,
                    adaptive_bvh: false,
                    ..Default::default()
                };
                settings.use_bvh = false;
                r.reset();
                r.render(&settings, true);
                let reference = r.read_pixels().unwrap();
                settings.use_bvh = true;
                r.reset();
                r.render(&settings, true);
                let bvh = r.read_pixels().unwrap();
                assert_pixels(
                    &reference,
                    &bvh,
                    &format!("seed={seed}, mode={mode:?}, p={p}, spatial={spatial}"),
                );
                settings.adaptive_bvh = true;
                r.reset();
                r.render(&settings, true);
                assert_pixels(&reference, &r.read_pixels().unwrap(), "adaptive traversal");
            }
        }
    }
    let s = Settings {
        spp: 8,
        ..Default::default()
    };
    r.reset();
    for _ in 0..4 {
        r.render(&s, true);
    }
    let reference = r.read_pixels().unwrap();
    let reference_png =
        std::env::temp_dir().join(format!("multiworld-eager-{}.png", std::process::id()));
    let deferred_png =
        std::env::temp_dir().join(format!("multiworld-deferred-{}.png", std::process::id()));
    r.save_png(&reference_png).unwrap();
    r.reset();
    for _ in 0..4 {
        r.accumulate(&s);
    }
    r.render(&s, false);
    assert_pixels(
        &reference,
        &r.read_pixels().unwrap(),
        "deferred presentation",
    );
    r.save_png(&deferred_png).unwrap();
    assert_eq!(
        std::fs::read(&reference_png).unwrap(),
        std::fs::read(&deferred_png).unwrap()
    );
    let _ = std::fs::remove_file(reference_png);
    let _ = std::fs::remove_file(deferred_png);
    for view in View::ALL {
        let s = Settings { view, ..s.clone() };
        r.render(&s, false);
        r.render(&s, false);
        assert_eq!(r.samples, 32);
        assert_pixels(
            &reference,
            &r.read_pixels().unwrap(),
            "paused views preserve sums",
        );
    }
    r.reset();
    r.render(&s, false);
    r.render(&s, false);
    assert!(
        r.read_pixels()
            .unwrap()
            .iter()
            .all(|p| bytemuck::bytes_of(p).iter().all(|b| *b == 0))
    );
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn spatial_primary_world_is_anchored_to_surface_not_orbiting_camera() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 16, 12, 42).unwrap();
    // Minimal repro: the same surface point seen from a camera orbit. The two
    // worlds' identical planes isolate primary selection from geometry/occlusion.
    let plane = object([0.; 3], 2., [0.5; 3]);
    r.set_worlds(&[vec![plane], vec![plane]]).unwrap();
    let mut params = Params::zeroed();
    params.transport = [5, 1, 42, 0];
    params.controls = [0.5, 0.8, 0., 7.];
    params.flags = [1, 0, 1, 1];
    params.acceleration = [1, 1, 1, 1];
    params.noise = [0.3, 0.35, 0.2, 3.];
    let mut rays = Vec::new();
    for angle in 0..24 {
        let theta = angle as f32 * std::f32::consts::TAU / 24.;
        let origin = Vec3::new(theta.sin() * 12., 5., theta.cos() * 12.);
        for seed in 0..1024 {
            let mut ray = probe(origin, (-origin).normalize(), 0);
            ray.direction[3] = seed as f32;
            rays.push(ray);
        }
    }
    let mut spatial_counts = Vec::new();
    for spatial in [false, true] {
        for policy in [0, 7] {
            params.flags[0] = u32::from(spatial);
            params.controls[3] = policy as f32;
            let result = dispatch_probes(&r, &rays, &params, "probe_primary_world");
            let counts: Vec<u32> = result
                .as_chunks::<1024>()
                .0
                .iter()
                .map(|angle| angle.iter().map(|p| p.bvh_t_normal[0] as u32).sum())
                .collect();
            let min = *counts.iter().min().unwrap();
            let max = *counts.iter().max().unwrap();
            eprintln!(
                "Primary A samples/1024 at one fixed point: spatial={spatial}, policy={policy}, range={min}..{max}"
            );
            if spatial {
                spatial_counts.push(counts);
            } else {
                assert_eq!(min, max, "Constant probability changed with camera angle");
            }
        }
    }
    assert_eq!(
        spatial_counts[0], spatial_counts[1],
        "Spatial bug differs between linear and BVH queries"
    );
    let min = *spatial_counts[0].iter().min().unwrap();
    let max = *spatial_counts[0].iter().max().unwrap();
    assert!(
        max - min <= 1,
        "Orbiting changed primary selection at a fixed surface point: {min}..{max} A samples/1024"
    );
    // The fix must preserve a spatial field, rather than replacing it with a
    // constant camera-independent mixture everywhere.
    let mut field_rays = Vec::new();
    let origin = Vec3::new(0., 8., 18.);
    for z in [-8., 0., 8.] {
        for x in [-8., 0., 8.] {
            let target = Vec3::new(x, 0., z);
            for seed in 0..1024 {
                let mut ray = probe(origin, (target - origin).normalize(), 0);
                ray.direction[3] = seed as f32;
                field_rays.push(ray);
            }
        }
    }
    let field = dispatch_probes(&r, &field_rays, &params, "probe_primary_world");
    let counts: Vec<u32> = field
        .as_chunks::<1024>()
        .0
        .iter()
        .map(|point| point.iter().map(|p| p.bvh_t_normal[0] as u32).sum())
        .collect();
    assert!(
        counts.iter().max().unwrap() - counts.iter().min().unwrap() > 256,
        "Spatial probability collapsed to a constant: {counts:?}"
    );
}

fn read_history(r: &Renderer) -> Vec<super::denoise::Frame> {
    let d = r.denoiser.as_ref().unwrap();
    let buffer = &d.history[d.current];
    let staging = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = r.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    r.read_buffer(encoder, &staging)
        .unwrap()
        .as_chunks::<{ size_of::<super::denoise::Frame>() }>()
        .0
        .iter()
        .map(|p| bytemuck::pod_read_unaligned(p))
        .collect()
}
#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn temporal_denoiser_preserves_raw_samples_and_rejects_wrong_world_history() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 48, 32, 42).unwrap();
    for mode in Mode::ALL {
        let mut s = Settings {
            mode,
            spp: 8,
            ..Default::default()
        };
        r.reset();
        r.render(&s, true);
        r.render(&s, true);
        let raw = r.read_pixels().unwrap();
        s.denoise = true;
        r.reset();
        r.render(&s, true);
        r.render(&s, true);
        assert_pixels(
            &raw,
            &r.read_pixels().unwrap(),
            "denoiser must not modify raw accumulators",
        );
        let history = read_history(&r);
        assert!(
            history
                .iter()
                .all(|p| p.sample.iter().all(|v| v.is_finite() && *v >= 0.))
        );
        assert!(history.iter().all(|p| (p.sample[3] - 16.).abs() < 0.001));
        r.render(&s, false);
        r.render(&s, false);
        assert!(
            read_history(&r)
                .iter()
                .zip(&history)
                .all(|(a, b)| a.sample == b.sample)
        );
    }
    let mut s = Settings {
        denoise: true,
        spp: 8,
        ..Default::default()
    };
    r.reset();
    r.render(&s, true);
    // Corrupt ONLY the B guide, while A and all shading data remain valid.
    // A single-world validation would incorrectly accept this bright old history.
    let mut history = read_history(&r);
    for pixel in &mut history {
        pixel.position_b[0] += 100.;
        pixel.sample = [10., 0., 0., 100.];
    }
    let d = r.denoiser.as_ref().unwrap();
    r.queue
        .write_buffer(&d.history[d.current], 0, bytemuck::cast_slice(&history));
    r.render(&s, true);
    assert!(
        read_history(&r)
            .iter()
            .all(|p| (p.sample[3] - 16.).abs() < 0.001),
        "The B world's guide must invalidate history even when A still matches"
    );
    // Camera motion may reuse compatible history, but never its full old weight.
    s.orbit(0.01, 0.);
    r.render(&s, true);
    assert_eq!(r.samples, 8);
    let moved = read_history(&r);
    assert!(
        moved.iter().any(|p| p.sample[3] > 8.1),
        "No compatible history was reprojected"
    );
    assert!(moved.iter().all(|p| p.sample[3] <= 40.001));
    // Mixture/transport edits invalidate history. A changed p cannot retain old light.
    s.p = 0.2;
    r.render(&s, true);
    assert_eq!(r.samples, 8);
    assert!(
        read_history(&r)
            .iter()
            .all(|p| (p.sample[3] - 8.).abs() < 0.001)
    );
    r.resize(32, 24).unwrap();
    r.render(&s, true);
    assert_eq!(read_history(&r).len(), 32 * 24);
    s.denoise_strength = 0.;
    r.render(&s, false);
    let temp = std::env::temp_dir();
    let a = temp.join(format!("multiworld-filter-zero-{}.png", std::process::id()));
    let b = temp.join(format!("multiworld-raw-{}.png", std::process::id()));
    r.save_png(&a).unwrap();
    r.save_raw_png(&b).unwrap();
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
    let _ = std::fs::remove_file(a);
    let _ = std::fs::remove_file(b);
}
