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
@compute @workgroup_size(64)
fn probe_corrected_guides(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&probes) {return;}
    var p=probes[id.x];let result=trace(p.origin.xyz,p.direction.xyz,u32(p.direction.w),u32(p.origin.w));
    p.brute_color=vec4<f32>(result.guide_albedo,1.0);p.bvh_t_normal=vec4<f32>(result.guide_normal,1.0);probes[id.x]=p;
}

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
fn probe_radiance(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&probes) {return;}
    let p=probes[id.x];
    let result=trace(p.origin.xyz,p.direction.xyz,hash(u32(p.direction.w)),params.transport.x);
    probes[id.x].brute_color=vec4<f32>(result.radiance,0.0);
}
@compute @workgroup_size(32)
fn probe_matrix(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    neural_pending=frame_store.neural.samples[id.x].label.w>0.0;
    neural_pending_sample=frame_store.neural.samples[id.x];
    let prediction=neural_resolve_matrix(lane);
    probes[id.x].brute_color=vec4<f32>(prediction,0.0);
}
@compute @workgroup_size(64)
fn probe_neural(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&probes) {return;}
    probes[id.x].brute_color=vec4<f32>(neural_predict(frame_store.neural.samples[id.x]),0.0);
}
@compute @workgroup_size(64)
fn probe_cached_radiance(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&probes) {return;}
    let p=probes[id.x];
    let result=trace(p.origin.xyz,p.direction.xyz,hash(u32(p.direction.w)),params.transport.x);
    probes[id.x].brute_color=vec4<f32>(result.radiance,0.0);
}
@compute @workgroup_size(64)
fn probe_primary_world(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&probes) {return;}
    let p=probes[id.x];
    let result=trace(p.origin.xyz,p.direction.xyz,u32(p.direction.w),params.transport.x);
    probes[id.x].bvh_t_normal=vec4<f32>(result.stats,0.0);
}
@compute @workgroup_size(64)
fn probe_dielectric(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&probes) {return;}
    let p=probes[id.x].origin;
    probes[id.x].brute_color=vec4<f32>(dielectric_fresnel(p.x,p.y,p.z),0.0,0.0,0.0);
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
    params.lights[0] = r.worlds[0].light_count;
    params.lights[1] = r.worlds[1].light_count;
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
            ray_query::shader(
                &r.device,
                format!(
                    "{}\n{}\n{}\n{}\n{}\n{PROBE_SHADER}",
                    if entry == "probe_matrix" {
                        "enable wgpu_cooperative_matrix;"
                    } else {
                        ""
                    },
                    concat!(
                        concat!(
                            include_str!("../shaders/trace.wgsl"),
                            "\n",
                            include_str!("../shaders/adaptive.wgsl")
                        ),
                        "\n",
                        include_str!("../shaders/caustics_stub.wgsl")
                    ),
                    include_str!("../shaders/nrc.wgsl"),
                    if entry == "probe_matrix" {
                        include_str!("../shaders/nrc_queued.wgsl")
                    } else {
                        ""
                    },
                    if entry == "probe_matrix" {
                        include_str!("../shaders/nrc_matrix.wgsl")
                    } else {
                        include_str!("../shaders/nrc_scalar.wgsl")
                    }
                ),
            )
            .into(),
        ),
    });
    let pipeline = r
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &shader,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("NRC_ENABLED", f64::from(entry == "probe_cached_radiance"))],
                ..Default::default()
            },
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
        scene::selected_worlds(
            42,
            [
                scene::WorldPreset::SunkenTemple,
                scene::WorldPreset::BioluminescentGrove,
            ],
        ),
        scene::selected_worlds(
            42,
            [
                scene::WorldPreset::LunarObservatory,
                scene::WorldPreset::NeonDistrict,
            ],
        ),
        scene::selected_worlds(
            42,
            [
                scene::WorldPreset::GlassGallery,
                scene::WorldPreset::CrystalGarden,
            ],
        ),
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
        if r.hardware_ray_tracing_available() {
            assert_queries(&queries_with_policy(&r, &rays, 48));
        }
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
                    denoiser: crate::settings::DenoiserKind::Temporal,
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
        denoiser: crate::settings::DenoiserKind::Temporal,
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
            denoiser: crate::settings::DenoiserKind::Temporal,
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
        denoiser: crate::settings::DenoiserKind::Temporal,
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

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn light_sampling_preserves_energy_and_reduces_variance() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 1, 1, 42).unwrap();
    let floor = Object {
        center_radius: [0.; 4],
        extent_shape: [0., 0., 0., 2.],
        color_material: [0.8, 0.8, 0.8, 0.],
        emission: [0.; 4],
    };
    let sphere = Object {
        center_radius: [0., 3., 0., 0.5],
        extent_shape: [0.; 4],
        color_material: [0.; 4],
        emission: [2., 1., 0.5, 0.],
    };
    let samples = 32768;
    let probes: Vec<_> = (0..samples)
        .map(|i| {
            let mut p = probe(Vec3::Y, -Vec3::Y, 0);
            p.direction[3] = i as f32;
            p
        })
        .collect();
    // Exact on-axis sphere irradiance, including the integrator's origin offset.
    let fraction = (0.5_f64 / (3. - 0.002)).powi(2);
    let square_k = 1.0_f64 / (1.0_f64 + (2.5_f64 - 0.002).powi(2)).sqrt();
    let square_fraction = 4.0 / std::f64::consts::PI * square_k * square_k.atan();
    for (name, light, hybrid, mirror, depth, blocked, expected) in [
        (
            "box",
            Object {
                extent_shape: [1., 0.5, 1., 1.],
                ..sphere
            },
            false,
            false,
            2,
            false,
            std::array::from_fn(|c| {
                0.8 * ([0.045, 0.055, 0.075][c] * (1. - square_fraction)
                    + f64::from(sphere.emission[c]) * square_fraction)
            }),
        ),
        (
            "hybrid box",
            Object {
                extent_shape: [1., 0.5, 1., 1.],
                ..sphere
            },
            true,
            false,
            2,
            false,
            std::array::from_fn(|c| {
                0.8 * ([0.045, 0.055, 0.075][c] * (1. - 0.5 * square_fraction)
                    + f64::from(sphere.emission[c]) * 0.5 * square_fraction)
            }),
        ),
        (
            "overlapping proposals",
            sphere,
            true,
            false,
            2,
            false,
            std::array::from_fn(|c| {
                0.8 * ([0.045, 0.055, 0.075][c] * (1. - 0.5 * fraction)
                    + f64::from(sphere.emission[c]) * 0.5 * fraction)
            }),
        ),
        (
            "sphere",
            sphere,
            false,
            false,
            2,
            false,
            std::array::from_fn(|c| {
                0.8 * ([0.045, 0.055, 0.075][c] * (1. - fraction)
                    + f64::from(sphere.emission[c]) * fraction)
            }),
        ),
        (
            "hybrid sphere",
            sphere,
            true,
            false,
            2,
            false,
            std::array::from_fn(|c| {
                0.8 * ([0.045, 0.055, 0.075][c] * (1. - 0.5 * fraction)
                    + f64::from(sphere.emission[c]) * 0.5 * fraction)
            }),
        ),
        (
            "inside sphere",
            Object {
                center_radius: [0., 0., 0., 4.],
                ..sphere
            },
            false,
            false,
            2,
            false,
            [1.6, 0.8, 0.4],
        ),
        (
            "inside box",
            Object {
                center_radius: [0.; 4],
                extent_shape: [4., 4., 4., 1.],
                ..sphere
            },
            false,
            false,
            2,
            false,
            [1.6, 0.8, 0.4],
        ),
        ("mirror", sphere, false, true, 2, false, [1.6, 0.8, 0.4]),
        ("depth truncation", sphere, false, false, 1, false, [0.; 3]),
        ("occluded", sphere, false, false, 2, true, [0.; 3]),
    ] {
        let mut floor = floor;
        floor.color_material[3] = if mirror { 1. } else { 0. };
        let mut a = vec![floor];
        let mut b = if hybrid {
            vec![light]
        } else {
            a.push(light);
            vec![]
        };
        if name == "overlapping proposals" {
            b.push(Object {
                center_radius: [0., 3., 0., 0.25],
                ..sphere
            });
        }
        if blocked {
            a.push(Object {
                center_radius: [0., 1.5, 0., 0.],
                extent_shape: [0., 0., 0., 2.],
                color_material: [0.; 4],
                emission: [0.; 4],
            });
        }
        r.set_worlds(&[a, b]).unwrap();
        let mut variances = [0.; 2];
        for enabled in [false, true] {
            let mut params = Params::zeroed();
            params.transport = [if hybrid { 4 } else { 0 }, depth, 42, 0];
            params.controls = [0.5, 0.8, 0., 7.];
            params.flags = [
                0,
                u32::from(hybrid),
                r.worlds[0].object_count,
                r.worlds[1].object_count,
            ];
            params.acceleration = [
                r.worlds[0].node_count,
                r.worlds[1].node_count,
                r.worlds[0].plane_count,
                r.worlds[1].plane_count,
            ];
            params.lights = [
                r.worlds[0].light_count,
                r.worlds[1].light_count,
                u32::from(enabled),
                0,
            ];
            let out = dispatch_probes(&r, &probes, &params, "probe_radiance");
            let means: [f64; 3] = std::array::from_fn(|c| {
                out.iter().map(|p| f64::from(p.brute_color[c])).sum::<f64>() / f64::from(samples)
            });
            for c in 0..3 {
                assert!(
                    (means[c] - expected[c]).abs() < 0.008,
                    "{name} MIS={enabled}: {means:?} expected {expected:?}"
                );
            }
            variances[usize::from(enabled)] = out
                .iter()
                .map(|p| (f64::from(p.brute_color[0]) - means[0]).powi(2))
                .sum::<f64>()
                / f64::from(samples);
            eprintln!(
                "{name}, MIS={enabled}: mean={means:?}, red variance={}",
                variances[usize::from(enabled)]
            );
        }
        if name == "sphere" {
            assert!(
                variances[1] * 10. < variances[0],
                "MIS must reduce sphere-light variance"
            );
        }
    }
    r.set_worlds(&[vec![floor], vec![floor]]).unwrap();
    for (p, expected_a) in [(0.25, 8.), (0.5, 16.), (0.75, 24.)] {
        let mut settings = Settings {
            mode: Mode::PerPath,
            p,
            depth: 2,
            spp: 32,
            denoiser: crate::settings::DenoiserKind::Temporal,
            ..Default::default()
        };
        r.reset();
        r.accumulate(&settings);
        let full = r.read_pixels().unwrap();
        assert_eq!(
            full[0].stats_a[0], expected_a,
            "stratified initial-world count"
        );
        r.reset();
        settings.spp = 7;
        r.accumulate(&settings);
        settings.spp = 25;
        r.accumulate(&settings);
        assert_pixels(
            &full,
            &r.read_pixels().unwrap(),
            "stratified irregular batches",
        );
    }
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn curved_geometry_matches_cpu_and_world_selection_rebuilds_atomically() {
    use crate::{geometry, scene::WorldPreset};
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 16, 12, 42).unwrap();
    let mut capsule = object([0., 0., 0.], 3., [0.5; 3]);
    capsule.center_radius[3] = 0.5;
    capsule.extent_shape = [0.5, 1.5, -0.3, 3.];
    let mut ellipse = object([0., 0., 0.], 4., [0.5; 3]);
    ellipse.extent_shape = [2., 0.4, 0.8, 4.];
    let scenes = [vec![capsule], vec![ellipse]];
    r.set_worlds(&scenes).unwrap();
    let rays = sample_rays(&scenes);
    let out = queries(&r, &rays);
    assert_queries(&out);
    for p in &out {
        let origin = Vec3::from_slice(&p.origin[..3]);
        let direction = Vec3::from_slice(&p.direction[..3]);
        let expected = if p.origin[3] == 0. {
            let half = Vec3::from_slice(&capsule.extent_shape[..3]);
            geometry::capsule(origin, direction, -half, half, 0.5)
        } else {
            geometry::ellipsoid(origin, direction, Vec3::ZERO, Vec3::new(2., 0.4, 0.8))
        };
        if let Some(mut hit) = expected {
            if hit.normal.dot(direction) > 0. {
                hit.normal = -hit.normal;
            }
            assert!(
                (p.bvh_t_normal[0] - hit.t).abs() < 0.002,
                "curved t: {:?} vs {}",
                p.bvh_t_normal,
                hit.t
            );
            assert!(Vec3::from_slice(&p.bvh_t_normal[1..]).abs_diff_eq(hit.normal, 0.005));
        } else {
            assert!(p.bvh_t_normal[0] > 1e29);
        }
    }
    let mut s = Settings {
        worlds: [WorldPreset::SunkenTemple, WorldPreset::BioluminescentGrove],
        spp: 2,
        denoise: true,
        denoiser: crate::settings::DenoiserKind::Temporal,
        ..Default::default()
    };
    r.render(&s, true);
    assert_eq!(r.scene_presets, s.worlds);
    assert_eq!(r.samples, 2);
    s.worlds[1] = WorldPreset::NeonDistrict;
    r.render(&s, false);
    assert_eq!(r.scene_presets, s.worlds);
    assert_eq!(r.samples, 0);
    let generated = scene::selected_worlds(s.seed, s.worlds);
    assert_eq!(r.worlds[1].object_count, generated[1].len() as u32);
    r.render(&s, true);
    let restarted = r.read_pixels().unwrap();
    let mut fresh = Renderer::new(r.device.clone(), r.queue.clone(), 16, 12, 42).unwrap();
    fresh.render(&s, true);
    assert_pixels(
        &restarted,
        &fresh.read_pixels().unwrap(),
        "preset change discards old accumulation",
    );
    // Explicit uploaded scenes persist until a seed or preset selection changes.
    r.set_worlds(&scenes).unwrap();
    r.render(&s, true);
    assert_eq!(r.worlds[0].object_count, 1);
    s.seed = 43;
    r.render(&s, true);
    assert_eq!(r.scene_seed, 43);
    assert_eq!(r.scene_presets, s.worlds);
    assert!(r.worlds[0].object_count > 1);
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NeuralSample {
    position: [f32; 4],
    normal: [f32; 4],
    albedo: [f32; 4],
    label: [f32; 4],
}
fn read_neural_header(r: &Renderer) -> [u32; 4] {
    let staging = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = r.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&r.frames, 0, &staging, 0, 16);
    bytemuck::pod_read_unaligned(&r.read_buffer(encoder, &staging).unwrap())
}
#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn fused_neural_training_fits_world_conditioned_held_out_targets() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 8, 8, 42).unwrap();
    let mut neural = nrc::Neural::new(
        &r.device,
        &r.shader,
        &r.layout,
        r.width,
        r.height,
        r.trace_group_y,
    );
    let mut encoder = r.device.create_command_encoder(&Default::default());
    neural.initialize(&mut encoder, &r.bind_group);
    r.queue.submit([encoder.finish()]);
    let samples: Vec<NeuralSample> = (0..1024)
        .map(|i| {
            let world = i % 2;
            NeuralSample {
                position: [(i as f32 / 1024. - 0.5) * 0.4, 0.002, 0., world as f32],
                normal: [0., 1., 0., 3.],
                albedo: [0.8, 0.8, 0.8, 0.],
                label: if world == 0 {
                    [1.0, 0.1, 0.02, 1.]
                } else {
                    [0.02, 0.1, 1.0, 1.]
                },
            }
        })
        .collect();
    r.queue.write_buffer(
        &r.frames,
        nrc::SAMPLE_OFFSET,
        bytemuck::cast_slice(&samples),
    );
    let mut params = Params::zeroed();
    params.transport = [4, 4, 42, 0];
    params.controls = [0.5, 0.8, 0., 0.];
    r.queue
        .write_buffer(&r.uniform, 0, bytemuck::bytes_of(&params));
    let pipeline = r
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fused supervised training test"),
            layout: Some(&r.pipeline_layout),
            module: &r.shader,
            entry_point: Some("neural_train_fused"),
            compilation_options: Default::default(),
            cache: None,
        });
    let make_phase = |entry| {
        r.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&r.pipeline_layout),
                module: &r.shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
    };
    let apply = make_phase("neural_apply_gradients");
    let finish = make_phase("neural_finish_training");
    let mut encoder = r.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &r.bind_group, &[]);
        for _ in 0..512 {
            pass.set_pipeline(&pipeline);
            pass.dispatch_workgroups(nrc::BATCH, 1, 1);
            pass.set_pipeline(&apply);
            pass.dispatch_workgroups(nrc::PARAMETERS.div_ceil(64), 1, 1);
            pass.set_pipeline(&finish);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }
    r.queue.submit([encoder.finish()]);
    assert_eq!(read_neural_header(&r)[0], 512 * nrc::BATCH);
    assert_eq!(read_neural_header(&r)[1], 512);
    // New coordinates, absent from training; the two worlds have opposite color.
    let held: Vec<NeuralSample> = (0..32)
        .map(|i| NeuralSample {
            position: [0.013 + (i / 2) as f32 * 0.002, 0.002, 0.01, (i % 2) as f32],
            ..samples[i % 2]
        })
        .collect();
    r.queue
        .write_buffer(&r.frames, nrc::SAMPLE_OFFSET, bytemuck::cast_slice(&held));
    let probes = vec![Probe::zeroed(); held.len()];
    let predictions = dispatch_probes(&r, &probes, &params, "probe_neural");
    let error = predictions
        .iter()
        .zip(&held)
        .flat_map(|(p, s)| (0..3).map(move |c| (p.brute_color[c] - s.label[c]).powi(2)))
        .sum::<f32>()
        / (held.len() * 3) as f32;
    eprintln!("Held-out world-conditioned neural RGB MSE: {error}");
    assert!(error < 0.015, "fused optimizer must fit both world targets");
    assert!(
        predictions
            .iter()
            .all(|p| p.brute_color.iter().all(|x| x.is_finite()))
    );
    r.neural = Some(neural);
    let mut s = Settings {
        neural_cache: true,
        spp: 1,
        denoiser: crate::settings::DenoiserKind::Temporal,
        ..Default::default()
    };
    r.render(&s, false);
    let before = read_neural_header(&r)[0];
    s.yaw += 0.1;
    r.render(&s, false);
    assert_eq!(
        read_neural_header(&r)[0],
        before,
        "camera retains world-space model"
    );
    s.denoise = true;
    r.render(&s, false);
    assert_eq!(
        read_neural_header(&r)[0],
        before,
        "guide reallocation preserves model prefix"
    );
    s.p = 0.25;
    r.render(&s, false);
    assert_eq!(
        read_neural_header(&r)[0],
        0,
        "transport edits clear model and optimizer"
    );
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn neural_termination_accounts_for_emission_depth_and_delta_surfaces() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 1, 1, 42).unwrap();
    let plane = |y, emission, metal| Object {
        center_radius: [0., y, 0., 0.],
        extent_shape: [0., 0., 0., 2.],
        color_material: [0.5, 0.5, 0.5, metal],
        emission: [emission, emission, emission, 0.],
    };
    let mut params = Params::zeroed();
    params.controls = [0.5, 0.8, 0., 0.];
    params.flags = [0, 0, 2, 0];
    params.neural = [1, 1, 0, 0];
    let ray = probe(Vec3::Y, -Vec3::Y, 0);
    for metal in [0., 1.] {
        r.set_worlds(&[vec![plane(0., 0., metal), plane(2., 0.1, metal)], vec![]])
            .unwrap();
        let mut weights = vec![0_f32; nrc::PARAMETERS as usize];
        weights[2208..2211].fill(2.);
        r.queue
            .write_buffer(&r.frames, 0, bytemuck::cast_slice(&[512_u32, 32, 0, 0]));
        r.queue
            .write_buffer(&r.frames, 16, bytemuck::cast_slice(&weights));
        for depth in [1, 2, 4] {
            params.transport = [0, depth, 42, 0];
            let output = dispatch_probes(&r, &[ray], &params, "probe_cached_radiance");
            let expected = if depth == 1 {
                0.
            } else if depth == 2 || metal > 0.5 {
                if depth == 2 { 0.05 } else { 0.0625 }
            } else {
                0.55
            };
            for c in 0..3 {
                assert!(
                    (output[0].brute_color[c] - expected).abs() < 1e-5,
                    "metal={metal}, depth={depth}, output={:?}, expected={expected}",
                    output[0].brute_color
                );
            }
        }
    }
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn neural_teachers_store_uncached_suffixes_without_camera_throughput() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 8, 8, 42).unwrap();
    let floor = Object {
        center_radius: [0.; 4],
        extent_shape: [0., 0., 0., 2.],
        color_material: [0.8, 0.8, 0.8, 0.],
        emission: [0.; 4],
    };
    let ceiling = Object {
        center_radius: [0., 2., 0., 0.],
        extent_shape: [0., 0., 0., 2.],
        color_material: [0.; 4],
        emission: [2., 1., 0.5, 0.],
    };
    r.set_worlds(&[vec![floor], vec![ceiling]]).unwrap();
    let settings = Settings {
        neural_cache: true,
        primary_a: true,
        depth: 2,
        spp: 1,
        denoiser: crate::settings::DenoiserKind::Temporal,
        ..Default::default()
    };
    for _ in 0..8 {
        r.accumulate(&settings);
    }
    let size = 1024 * size_of::<NeuralSample>() as u64;
    let staging = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = r.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&r.frames, nrc::SAMPLE_OFFSET, &staging, 0, size);
    let data = r.read_buffer(encoder, &staging).unwrap();
    let samples: Vec<NeuralSample> = data
        .as_chunks::<64>()
        .0
        .iter()
        .map(|v| bytemuck::pod_read_unaligned(v))
        .collect();
    let mut a = 0;
    let mut b = 0;
    for (i, sample) in samples.iter().enumerate() {
        if i % 4 != 0 {
            assert_eq!(sample.label[3], 0.);
            continue;
        }
        assert_eq!(sample.label[3], 1.);
        assert_eq!(sample.position[3], 0.);
        assert_eq!(sample.normal[3], 1.);
        if (sample.label[0] - 2.).abs() < 1e-5 {
            b += 1;
            for (actual, expected) in sample.label[..3].iter().zip([2., 1., 0.5]) {
                assert!((actual - expected).abs() < 1e-5);
            }
        } else {
            a += 1;
            for (actual, expected) in sample.label[..3].iter().zip([0.045, 0.055, 0.075]) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }
    assert!(
        a > 50 && b > 50,
        "both independent world continuations must be sampled: {a}, {b}"
    );
    assert!(read_neural_header(&r)[0] > 0);
}

#[test]
#[ignore = "diagnostic: requires Metal/Vulkan/DX12 compute adapter"]
fn neural_gpu_phase_profile() {
    use std::time::Instant;
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 160, 120, 42).unwrap();
    let settings = Settings {
        neural_cache: true,
        worlds: [
            scene::WorldPreset::SunkenTemple,
            scene::WorldPreset::BioluminescentGrove,
        ],
        spp: 1,
        cache_bounce: 2,
        denoiser: crate::settings::DenoiserKind::Temporal,
        ..Default::default()
    };
    for _ in 0..128 {
        r.accumulate(&settings);
    }
    let pipeline = |entry, enabled| {
        r.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&r.pipeline_layout),
                module: &r.shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("NRC_ENABLED", enabled),
                        ("NRC_TEACHER", f64::from(entry == "neural_teach")),
                        ("ACTIVE_MODE", 4.),
                    ],
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            })
    };
    let base = pipeline("accumulate", 0.);
    let cached = pipeline("accumulate", 1.);
    let teach = pipeline("neural_teach", 1.);
    let train = pipeline("neural_train_serial_reference", 1.);
    let gradient = pipeline("neural_train_fused", 1.);
    let apply = pipeline("neural_apply_gradients", 1.);
    let wait = || {
        r.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(60)),
            })
            .unwrap()
    };
    wait();
    let trained = read_neural_header(&r)[0];
    for (name, pipe, groups, ready) in [
        ("ordinary", &base, [20, 15, 1], true),
        ("nrc-no-query", &cached, [20, 15, 1], false),
        ("nrc-frozen", &cached, [20, 15, 1], true),
        (
            "queued-trace",
            r.neural.as_ref().unwrap().trace_pipeline(4, false),
            [20, 15, 1],
            true,
        ),
        (
            "matrix-inference",
            r.neural.as_ref().unwrap().resolve_pipeline(4),
            [600, 1, 1],
            true,
        ),
        ("teachers", &teach, [4, 1, 1], true),
        ("training-reference", &train, [1, 1, 1], true),
        ("gradient-minibatch", &gradient, [128, 1, 1], true),
        ("apply-minibatch", &apply, [35, 1, 1], true),
    ] {
        r.queue.write_buffer(
            &r.frames,
            0,
            bytemuck::bytes_of(&if ready { trained } else { 0_u32 }),
        );
        wait();
        let mut durations = Vec::new();
        for _ in 0..3 {
            let mut encoder = r.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, &r.bind_group, &[]);
                pass.set_bind_group(1, &r.neural.as_ref().unwrap().queries, &[]);
                for _ in 0..64 {
                    pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
                }
            }
            let start = Instant::now();
            r.queue.submit([encoder.finish()]);
            wait();
            durations.push(start.elapsed().as_secs_f64() * 1000. / 64.);
        }
        durations.sort_by(f64::total_cmp);
        eprintln!("NRC_PROFILE {name}: {:.4} ms/dispatch", durations[1]);
    }
}

#[test]
#[ignore = "requires Metal cooperative matrix adapter"]
fn cooperative_neural_inference_matches_scalar_and_handles_partial_tiles() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    if !device
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    {
        return;
    }
    let mut r = Renderer::new(device, queue, 28, 19, 42).unwrap();
    let mut neural = nrc::Neural::new(
        &r.device,
        &r.shader,
        &r.layout,
        r.width,
        r.height,
        r.trace_group_y,
    );
    let mut encoder = r.device.create_command_encoder(&Default::default());
    neural.initialize(&mut encoder, &r.bind_group);
    r.queue.submit([encoder.finish()]);
    r.queue.write_buffer(
        &r.frames,
        16 + 2208 * 4,
        bytemuck::cast_slice(&[1_f32, 0.7, 0.3]),
    );
    let mut rng = Rng(137);
    let samples: Vec<_> = (0..32)
        .map(|i| NeuralSample {
            position: [
                rng.unit() * 8. - 4.,
                rng.unit() * 5.,
                rng.unit() * 8. - 4.,
                (i % 2) as f32,
            ],
            normal: [0., 1., 0., 4.],
            albedo: [0.8, 0.4, 0.1, 0.],
            label: [0., 0., 0., if i % 5 == 0 { 0. } else { 1. }],
        })
        .collect();
    r.queue.write_buffer(
        &r.frames,
        nrc::SAMPLE_OFFSET,
        bytemuck::cast_slice(&samples),
    );
    let mut params = Params::zeroed();
    params.controls = [0.35, 0.8, 0., 0.];
    params.transport = [4, 5, 42, 0];
    let probes = vec![Probe::zeroed(); 32];
    let scalar = dispatch_probes(&r, &probes, &params, "probe_neural");
    let matrix = dispatch_probes(&r, &probes, &params, "probe_matrix");
    let mut error = 0_f32;
    for (i, (a, b)) in scalar.iter().zip(&matrix).enumerate() {
        if samples[i].label[3] > 0. {
            for c in 0..3 {
                error = error.max((a.brute_color[c] - b.brute_color[c]).abs());
            }
        }
    }
    eprintln!("Cooperative/scalar max neural RGB error: {error}");
    assert!(error < 0.0001);
    r.queue
        .write_buffer(&r.frames, 0, bytemuck::cast_slice(&[512_u32, 32, 0, 0]));
    r.neural = Some(neural);
    r.render(
        &Settings {
            neural_cache: true,
            cache_bounce: 1,
            spp: 2,
            denoiser: crate::settings::DenoiserKind::Temporal,
            ..Default::default()
        },
        true,
    );
    let pixels = r.read_pixels().unwrap();
    assert_eq!(pixels.len(), 28 * 19);
    assert!(
        bytemuck::cast_slice::<Pixel, f32>(&pixels)
            .iter()
            .all(|v| v.is_finite())
    );
}

#[test]
#[ignore = "requires Metal cooperative matrix adapter"]
fn queued_neural_render_matches_scalar_in_all_world_modes_and_guides() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    if !device
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    {
        return;
    }
    let mut r = Renderer::new(device, queue, 28, 19, 42).unwrap();
    let mut neural = nrc::Neural::new(
        &r.device,
        &r.shader,
        &r.layout,
        r.width,
        r.height,
        r.trace_group_y,
    );
    let mut encoder = r.device.create_command_encoder(&Default::default());
    neural.initialize(&mut encoder, &r.bind_group);
    r.queue.submit([encoder.finish()]);
    r.queue.write_buffer(
        &r.frames,
        16 + 2208 * 4,
        bytemuck::cast_slice(&[1_f32, 0.7, 0.3]),
    );
    let staging = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(nrc::PARAMETERS) * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = r.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&r.frames, 16, &staging, 0, u64::from(nrc::PARAMETERS) * 4);
    let weights = r.read_buffer(encoder, &staging).unwrap();
    r.neural = Some(neural);
    for guides in [false, true] {
        for mode in Mode::ALL {
            let settings = Settings {
                mode,
                neural_cache: true,
                cache_bounce: 1,
                depth: 8,
                spp: 1,
                denoise: guides,
                denoiser: crate::settings::DenoiserKind::Temporal,
                ..Default::default()
            };
            r.previous = Some(settings.clone());
            r.clear_accumulation();
            r.queue.write_buffer(&r.frames, 16, &weights);
            r.queue
                .write_buffer(&r.frames, 0, bytemuck::cast_slice(&[512_u32, 32, 0, 0]));
            r.render(&settings, true);
            let queued = r.read_pixels().unwrap();
            let read_guides = |r: &Renderer| {
                let size = u64::from(r.width)
                    * u64::from(r.height)
                    * size_of::<super::denoise::Frame>() as u64;
                let staging = r.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let mut encoder = r.device.create_command_encoder(&Default::default());
                encoder.copy_buffer_to_buffer(&r.frames, nrc::BYTES, &staging, 0, size);
                r.read_buffer(encoder, &staging).unwrap()
            };
            let queued_guides = guides.then(|| read_guides(&r));
            // Replay exactly the same camera/path samples using a frozen scalar MLP.
            // params.size.w remains zero, so this replaces the image rather than adds.
            r.queue.write_buffer(&r.frames, 16, &weights);
            let pipeline = r
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Scalar neural rendering reference"),
                    layout: Some(&r.pipeline_layout),
                    module: &r.shader,
                    entry_point: Some("accumulate"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[
                            ("NRC_ENABLED", 1.),
                            ("ACTIVE_MODE", mode as u32 as f64),
                            ("DENOISE_INPUT", f64::from(guides)),
                        ],
                        ..Default::default()
                    },
                    cache: None,
                });
            let mut encoder = r.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &r.bind_group, &[]);
                pass.dispatch_workgroups(r.width.div_ceil(8), r.height.div_ceil(8), 1);
            }
            r.queue.submit([encoder.finish()]);
            let scalar = r.read_pixels().unwrap();
            let error = bytemuck::cast_slice::<Pixel, f32>(&queued)
                .iter()
                .zip(bytemuck::cast_slice::<Pixel, f32>(&scalar))
                .map(|(a, b)| (a - b).abs())
                .fold(0_f32, f32::max);
            assert!(
                error < 0.0001,
                "{mode:?} guides={guides}: queued/scalar error {error}"
            );
            if let Some(queued_guides) = queued_guides {
                let scalar_guides = read_guides(&r);
                let error = queued_guides
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(scalar_guides.as_chunks::<4>().0)
                    .map(|(a, b)| (f32::from_ne_bytes(*a) - f32::from_ne_bytes(*b)).abs())
                    .fold(0_f32, f32::max);
                assert!(error < 0.001, "{mode:?}: fresh guide error {error}");
                for (a, b) in queued_guides
                    .as_chunks::<112>()
                    .0
                    .iter()
                    .zip(scalar_guides.as_chunks::<112>().0)
                {
                    for (a, b) in a[..12]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(b[..12].as_chunks::<4>().0)
                    {
                        assert!(
                            (f32::from_ne_bytes(*a) - f32::from_ne_bytes(*b)).abs() < 0.0001,
                            "{mode:?}: fresh guide radiance mismatch"
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn neural_bootstrapped_teachers_keep_uncached_anchors_and_cold_start() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 8, 8, 42).unwrap();
    let plane = |y, albedo| Object {
        center_radius: [0., y, 0., 0.],
        extent_shape: [0., 0., 0., 2.],
        color_material: [albedo, albedo, albedo, 0.],
        emission: [0.; 4],
    };
    let room = vec![plane(0., 0.8), plane(3., 0.7)];
    r.set_worlds(&[room.clone(), room]).unwrap();
    let settings = Settings {
        mode: Mode::WorldA,
        neural_cache: true,
        cache_bounce: 1,
        cache_self_training: true,
        light_sampling: false,
        distance: 2.,
        depth: 8,
        spp: 1,
        denoiser: crate::settings::DenoiserKind::Temporal,
        ..Default::default()
    };
    r.render(&settings, false);
    // Constant raw output [3,1,.5] despite nonzero hidden/output weights.
    // Different EMA matrices catch selecting the wrong bank in either layer.
    let mut training_weights = vec![0_f32; nrc::PARAMETERS as usize];
    let mut rendering_weights = training_weights.clone();
    for i in 0..32 {
        training_weights[1024 + i] = 1.;
        training_weights[1056 + i * 32 + i] = 1.;
        training_weights[2112 + i * 3] = 1. / 32.;
        rendering_weights[i * 32 + i] = 0.2;
        rendering_weights[1056 + i * 32 + i] = 0.3;
    }
    training_weights[2208..].copy_from_slice(&[2., 1., 0.5]);
    rendering_weights[2208..].copy_from_slice(&[4., 3., 2.]);
    r.queue
        .write_buffer(&r.frames, 16, bytemuck::cast_slice(&rendering_weights));
    r.queue.write_buffer(
        &r.frames,
        nrc::TRAINING_OFFSET,
        bytemuck::cast_slice(&training_weights),
    );
    r.queue
        .write_buffer(&r.frames, 0, bytemuck::cast_slice(&[512_u32, 32, 0, 0]));
    let read_samples = |r: &Renderer| {
        let size = 1024 * size_of::<NeuralSample>() as u64;
        let staging = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = r.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&r.frames, nrc::SAMPLE_OFFSET, &staging, 0, size);
        r.read_buffer(encoder, &staging)
            .unwrap()
            .as_chunks::<64>()
            .0
            .iter()
            .map(|v| bytemuck::pod_read_unaligned::<NeuralSample>(v))
            .collect::<Vec<_>>()
    };
    r.accumulate(&settings);
    let samples = read_samples(&r);
    let mut anchors = 0;
    for path in samples.as_chunks::<4>().0 {
        if path[0].albedo[3] == 2. {
            anchors += 1;
            for s in path {
                assert_eq!(
                    s.label[..3],
                    [0.; 3],
                    "uncached anchors in a black room must not inherit the model's light"
                );
            }
        } else {
            assert_eq!(
                path[0].albedo[3], 1.,
                "ready non-anchor teachers must bootstrap"
            );
            for (s, factor) in path[..2].iter().zip([0.56, 0.8]) {
                assert_eq!(s.label[3], 1.);
                for (actual, prediction) in s.label[..3].iter().zip([3., 1., 0.5]) {
                    assert!(
                        (actual - factor * prediction).abs() < 0.00001,
                        "camera-prefix throughput or training weight selection was incorrect"
                    );
                }
            }
            assert_eq!(
                path[2].label[3], 0.,
                "the terminal query must not train on its own prediction"
            );
        }
    }
    assert_eq!(anchors, 16, "exactly 1/16 of 256 teachers stay uncached");
    // Replay the teacher pass using the portable scalar predictor with the
    // same raw training snapshot, independent of cooperative matrix support.
    r.queue.write_buffer(
        &r.frames,
        nrc::TRAINING_OFFSET,
        bytemuck::cast_slice(&training_weights),
    );
    let pipeline = r
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Scalar teacher reference"),
            layout: Some(&r.pipeline_layout),
            module: &r.shader,
            entry_point: Some("neural_teach"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[
                    ("NRC_ENABLED", 1.),
                    ("NRC_TEACHER", 1.),
                    ("ACTIVE_MODE", 0.),
                ],
                ..Default::default()
            },
            cache: None,
        });
    let mut encoder = r.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &r.bind_group, &[]);
        pass.dispatch_workgroups(4, 1, 1);
    }
    r.queue.submit([encoder.finish()]);
    let scalar_samples = read_samples(&r);
    for (a, b) in samples.iter().zip(&scalar_samples) {
        for (a, b) in a.label.iter().zip(b.label) {
            assert!(
                (a - b).abs() < 0.00001,
                "batched and scalar self-training targets differ"
            );
        }
    }
    r.clear_accumulation();
    r.queue
        .write_buffer(&r.frames, 0, bytemuck::cast_slice(&[0_u32; 4]));
    r.accumulate(&settings);
    for s in read_samples(&r) {
        assert_ne!(
            s.albedo[3], 1.,
            "cold teachers must not bootstrap random weights"
        );
        assert_eq!(s.label[..3], [0.; 3]);
    }
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn glass_fresnel_beer_absorption_total_internal_reflection_and_mesh_queries() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 8, 8, 42).unwrap();
    let mut params = Params::zeroed();
    let probes: Vec<_> = [
        (1., 1., 1.5, 0.04),
        (0.5, 1.5, 1., 1.),
        (1., 1., 1., 0.),
        (0., 1., 1.5, 1.),
    ]
    .into_iter()
    .map(|(c, a, b, _)| Probe {
        origin: [c, a, b, 0.],
        ..Probe::zeroed()
    })
    .collect();
    let result = dispatch_probes(&r, &probes, &params, "probe_dielectric");
    for (p, expected) in result.iter().zip([0.04, 1., 0., 1.]) {
        assert!((p.brute_color[0] - expected).abs() < 1e-5);
    }
    params.transport = [0, 32, 42, 0];
    params.controls = [0.5, 0.8, 0., 1.];
    params.flags = [0, 0, 1, 0];
    params.acceleration = [1, 0, 0, 0];
    params.lights = [0, 0, 0, 1];
    let rays: Vec<_> = (0..8192)
        .map(|i| {
            let mut p = probe(Vec3::Z * 3., -Vec3::Z, 0);
            p.direction[3] = i as f32;
            p
        })
        .collect();
    for (tint, ior) in [
        ([1.; 3], 1.5_f32),
        ([0.8, 0.5, 0.9], 1.5),
        ([0.8, 0.5, 0.9], 1.),
    ] {
        let mut ball = object([0.; 3], 0., tint);
        ball.center_radius[3] = 1.;
        ball.color_material[3] = 2. + ior;
        r.set_worlds(&[vec![ball], vec![]]).unwrap();
        let result = dispatch_probes(&r, &rays, &params, "probe_radiance");
        for (c, background) in [0.045, 0.055, 0.075].into_iter().enumerate() {
            let a = tint[c].powf(1.998);
            let f = ((ior - 1.) / (ior + 1.)).powi(2);
            let expected = background * (f + (1. - f) * (1. - f) * a / (1. - f * a));
            let mean = result.iter().map(|p| p.brute_color[c]).sum::<f32>() / result.len() as f32;
            assert!(
                (mean - expected).abs() < 0.0005,
                "channel {c}: Beer/Fresnel {mean}, expected {expected}"
            );
        }
    }
    let mut ball = object([0.; 3], 0., [1.; 3]);
    ball.center_radius[3] = 1.;
    ball.color_material[3] = 3.5;
    r.set_worlds(&[vec![ball], vec![]]).unwrap();
    let trapped: Vec<_> = (0..128)
        .map(|i| {
            let mut p = probe(Vec3::Z * 0.9, Vec3::X, 0);
            p.direction[3] = i as f32;
            p
        })
        .collect();
    for p in dispatch_probes(&r, &trapped, &params, "probe_radiance") {
        assert_eq!(
            p.brute_color[..3],
            [0.; 3],
            "total internal reflection must not leak background"
        );
    }
    let triangles = scene::mesh::instance(
        scene::mesh::teapot(),
        glam::Mat4::IDENTITY,
        scene::GlassPreset::Clear.material(),
    );
    r.set_worlds(&[triangles.clone(), vec![]]).unwrap();
    let rays: Vec<_> = triangles
        .iter()
        .step_by(17)
        .map(|t| {
            let center = Vec3::from_slice(&t.center_radius[..3])
                + (Vec3::from_slice(&t.extent_shape[..3]) + Vec3::from_slice(&t.emission[..3]))
                    / 3.;
            let eye = Vec3::new(0., 5., 8.);
            probe(eye, (center - eye).normalize(), 0)
        })
        .collect();
    let hits = queries(&r, &rays);
    assert_queries(&hits);
    assert!(
        hits.iter().all(|p| p.brute_emission[..3] == [0.; 3]),
        "triangle edge data must never become emitted light"
    );
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn glass_lens_focuses_caustics_and_mesh_worlds_remain_finite() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 16, 12, 42).unwrap();
    let floor = Object {
        center_radius: [0.; 4],
        extent_shape: [0., 0., 0., 2.],
        color_material: [0.75, 0.75, 0.75, 0.],
        emission: [0.; 4],
    };
    let light = Object {
        center_radius: [0., 6.5, 0., 0.3],
        extent_shape: [0.; 4],
        color_material: [0.; 4],
        emission: [100., 100., 100., 0.],
    };
    let mut means = Vec::new();
    let rays: Vec<_> = (0..65536)
        .map(|i| {
            let mut p = probe(Vec3::Y * 0.03, -Vec3::Y, 0);
            p.direction[3] = i as f32;
            p
        })
        .collect();
    for ior in [1., 1.5] {
        let lens = Object {
            center_radius: [0., 1.6, 0., 0.9],
            extent_shape: [0.; 4],
            color_material: [1., 1., 1., 2. + ior],
            emission: [0.; 4],
        };
        r.set_worlds(&[vec![floor, lens, light], vec![]]).unwrap();
        let mut params = Params::zeroed();
        params.transport = [0, 12, 42, 0];
        params.controls = [0.5, 0.8, 0., 1.];
        params.flags = [0, 0, 3, 0];
        params.acceleration = [r.worlds[0].node_count, 0, r.worlds[0].plane_count, 0];
        params.lights = [r.worlds[0].light_count, 0, 1, 1];
        let result = dispatch_probes(&r, &rays, &params, "probe_radiance");
        means.push(
            result.iter().map(|p| p.brute_color[0] as f64).sum::<f64>() / result.len() as f64,
        );
    }
    eprintln!(
        "Floor radiance: index-matched {}, glass caustic {}",
        means[0], means[1]
    );
    assert!(
        means[1] > means[0] * 2.,
        "the glass lens must focus actual transported light"
    );
    let mut settings = Settings {
        worlds: [
            scene::WorldPreset::GlassGallery,
            scene::WorldPreset::CrystalGarden,
        ],
        glass: [scene::GlassPreset::Clear, scene::GlassPreset::Diamond],
        depth: 24,
        spp: 2,
        denoiser: crate::settings::DenoiserKind::Temporal,
        ..Default::default()
    };
    for mode in Mode::ALL {
        settings.mode = mode;
        r.render(&settings, true);
        assert_eq!(r.samples, 2);
        assert!(
            bytemuck::cast_slice::<Pixel, f32>(&r.read_pixels().unwrap())
                .iter()
                .all(|v| v.is_finite()),
            "{mode:?}: glass transport must stay finite"
        );
    }
    // Querying a neural diffuse cache must not replace dielectric interfaces.
    let ball = Object {
        center_radius: [0., 0., 0., 1.],
        extent_shape: [0.; 4],
        color_material: [1., 1., 1., 3.5],
        emission: [0.; 4],
    };
    r.set_worlds(&[vec![ball], vec![]]).unwrap();
    let mut weights = vec![0_f32; nrc::PARAMETERS as usize];
    weights[2208..].fill(10.);
    r.queue
        .write_buffer(&r.frames, 16, bytemuck::cast_slice(&weights));
    r.queue
        .write_buffer(&r.frames, 0, bytemuck::cast_slice(&[512_u32, 32, 0, 0]));
    let mut params = Params::zeroed();
    params.transport = [0, 24, 42, 0];
    params.flags = [0, 0, 1, 0];
    params.acceleration = [1, 0, 0, 0];
    params.controls = [0.5, 0.8, 0., 1.];
    params.neural = [1, 1, 0, 0];
    params.lights = [0, 0, 0, 1];
    let rays: Vec<_> = (0..4096)
        .map(|i| {
            let mut p = probe(Vec3::Z * 3., -Vec3::Z, 0);
            p.direction[3] = i as f32;
            p
        })
        .collect();
    let result = dispatch_probes(&r, &rays, &params, "probe_cached_radiance");
    let mean = result.iter().map(|p| p.brute_color[0]).sum::<f32>() / result.len() as f32;
    assert!(
        (mean - 0.045).abs() < 0.001,
        "glass incorrectly terminated into the diffuse network: {mean}"
    );
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn fast_caustics_partition_energy_blend_and_fallback() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 32, 24, 42).unwrap();
    let floor = Object {
        center_radius: [0.; 4],
        extent_shape: [0., 0., 0., 2.],
        color_material: [0.75, 0.75, 0.75, 0.],
        emission: [0.; 4],
    };
    let light = Object {
        center_radius: [0., 6.5, 0., 0.3],
        extent_shape: [0.; 4],
        color_material: [0.; 4],
        emission: [100., 100., 100., 0.],
    };
    let mut s = Settings {
        mode: Mode::WorldA,
        depth: 12,
        yaw: 0.,
        pitch: 0.7,
        distance: 8.,
        denoiser: crate::settings::DenoiserKind::Temporal,
        ..Default::default()
    };
    // Index-matched glass catches missing/doubled straight-through illumination;
    // the lens catches incorrect light-side eta^2 and camera splat Jacobians.
    for (ior, pane) in [(1., false), (1.5, false), (1.5, true)] {
        let lens = Object {
            center_radius: [0., 1.6, 0., 0.9],
            extent_shape: [0.; 4],
            color_material: [0.85, 0.95, 0.99, 2. + ior],
            emission: [0.; 4],
        };
        let mut world = vec![floor, lens, light];
        if pane {
            world.push(Object {
                center_radius: [0., 2., 2., 0.],
                extent_shape: [2., 1.5, 0.04, 1.],
                color_material: [0.95, 0.98, 1., 3.5],
                emission: [0.; 4],
            });
        }
        r.set_worlds(&[world, vec![]]).unwrap();
        let mut means = Vec::new();
        let mut images = Vec::new();
        for (enabled, samples) in [(false, 8192), (true, 128)] {
            s.caustics = enabled;
            r.reset();
            while r.samples < samples {
                s.spp = (samples - r.samples).min(16);
                r.accumulate(&s);
                r.device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: Some(Duration::from_secs(60)),
                    })
                    .unwrap();
            }
            let p = r.read_pixels().unwrap();
            assert!(
                bytemuck::cast_slice::<Pixel, f32>(&p)
                    .iter()
                    .all(|v| v.is_finite())
            );
            means.push(
                p.iter()
                    .map(|p| p.a[..3].iter().sum::<f32>() as f64 / r.samples as f64)
                    .sum::<f64>()
                    / (p.len() * 3) as f64,
            );
            images.push(
                p.iter()
                    .map(|p| (p.a[0] + p.a[1] + p.a[2]) as f64 / (3. * r.samples as f64))
                    .collect::<Vec<_>>(),
            );
        }
        let bright: Vec<_> = images[0]
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > 0.3)
            .map(|(i, _)| i)
            .collect();
        let a: f64 = bright.iter().map(|i| images[0][*i]).sum();
        let b: f64 = bright.iter().map(|i| images[1][*i]).sum();
        eprintln!(
            "IOR {ior}, pane {pane}: camera {}, photons {}; bright {} pixels, energy ratio {}",
            means[0],
            means[1],
            bright.len(),
            b / a
        );
        if ior > 1. {
            assert!(!bright.is_empty());
            assert!((b / a - 1.).abs() < 0.15, "caustic-region energy mismatch");
        }
        assert!(
            (means[1] / means[0] - 1.).abs() < 0.03,
            "photon energy differs from reference: {means:?}"
        );
    }
    // Both photon maps must route to the same channel as their independent
    // fixed-world estimates, including different emission and glass colors.
    let lens_a = Object {
        center_radius: [0., 1.6, 0., 0.9],
        extent_shape: [0.; 4],
        color_material: [0.8, 0.95, 0.99, 3.5],
        emission: [0.; 4],
    };
    let lens_b = Object {
        color_material: [0.99, 0.85, 0.7, 3.62],
        ..lens_a
    };
    let light_b = Object {
        emission: [60., 90., 130., 0.],
        ..light
    };
    r.set_worlds(&[vec![floor, lens_a, light], vec![floor, lens_b, light_b]])
        .unwrap();
    s.caustics = true;
    s.spp = 4;
    let mut fixed = Vec::new();
    for mode in [Mode::WorldA, Mode::WorldB] {
        s.mode = mode;
        r.reset();
        r.accumulate(&s);
        fixed.push(r.read_pixels().unwrap());
    }
    s.mode = Mode::Blend;
    r.reset();
    r.accumulate(&s);
    let combined = r.read_pixels().unwrap();
    for (i, p) in combined.iter().enumerate() {
        for c in 0..3 {
            assert!((p.a[c] - fixed[0][i].a[c]).abs() < 1e-4 * (1. + p.a[c].abs()));
            assert!((p.b[c] - fixed[1][i].a[c]).abs() < 1e-4 * (1. + p.b[c].abs()));
        }
    }
    s.mode = Mode::WorldA;
    // No specular surfaces: direct-light contributions must be bit-identical.
    r.set_worlds(&[vec![floor, light], vec![floor, light]])
        .unwrap();
    s.spp = 4;
    s.caustics = false;
    r.reset();
    r.accumulate(&s);
    let baseline = r.read_pixels().unwrap();
    s.caustics = true;
    r.reset();
    r.accumulate(&s);
    let fast = r.read_pixels().unwrap();
    for (a, b) in baseline.iter().zip(&fast) {
        assert_eq!(a.a, b.a);
    }
    for mode in [Mode::PerPath, Mode::PerBounce, Mode::Persistent] {
        s.mode = mode;
        s.caustics = false;
        r.reset();
        r.accumulate(&s);
        let a = r.read_pixels().unwrap();
        s.caustics = true;
        r.reset();
        r.accumulate(&s);
        let b = r.read_pixels().unwrap();
        assert_eq!(
            bytemuck::cast_slice::<Pixel, u8>(&a),
            bytemuck::cast_slice::<Pixel, u8>(&b),
            "mixed modes must use the reference"
        );
    }
    s.mode = Mode::Blend;
    s.denoise = true;
    s.neural_cache = true;
    r.reset();
    r.render(&s, true);
    let blend = r.read_pixels().unwrap();
    assert_eq!(
        r.neural_training_batches(),
        0,
        "NRC must not learn overlapping caustics while photons are active"
    );
    for p in blend {
        assert_eq!(p.a, p.b);
    }
    // Resize recreates all pixel splat storage, including odd-sized dispatches.
    r.resize(33, 25).unwrap();
    r.render(&s, true);
    assert!(
        bytemuck::cast_slice::<Pixel, f32>(&r.read_pixels().unwrap())
            .iter()
            .all(|v| v.is_finite())
    );
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn corrected_glass_guides_reach_diffuse_receiver_and_apply_absorption() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 8, 8, 42).unwrap();
    let lens = Object {
        center_radius: [0., 1.5, 0., 1.],
        extent_shape: [0.; 4],
        color_material: [0.8, 0.9, 1., 3.],
        emission: [0.; 4],
    };
    let floor = Object {
        center_radius: [0.; 4],
        extent_shape: [0., 0., 0., 2.],
        color_material: [0.4, 0.6, 0.8, 0.],
        emission: [0.; 4],
    };
    r.set_worlds(&[vec![lens, floor], vec![]]).unwrap();
    let mut params = Params::zeroed();
    params.transport = [0, 8, 42, 0];
    params.flags = [0, 0, 2, 0];
    params.lights = [0, 0, 1, 1];
    params.controls = [0.5, 0.8, 0., 1.];
    params.acceleration = [r.worlds[0].node_count, 0, r.worlds[0].plane_count, 0];
    let p = probe(Vec3::new(0.4, 4., 0.), -Vec3::Y, 0);
    let output = dispatch_probes(&r, &[p], &params, "probe_corrected_guides");
    assert_eq!(&output[0].bvh_t_normal[..3], &[0., 1., 0.]);
    let chord = 2. * (1_f32 - 0.4 * 0.4).sqrt();
    for c in 0..3 {
        let target = floor.color_material[c] * lens.color_material[c].powf(chord);
        assert!((output[0].brute_color[c] - target).abs() < 0.003);
    }
}

#[test]
#[ignore = "requires installed OIDN runtime and Metal GPU"]
fn oidn_export_preserves_raw_and_rejects_stale_scene_results() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 32, 24, 42).unwrap();
    let settings = Settings {
        mode: Mode::WorldA,
        denoise: true,
        denoiser: crate::settings::DenoiserKind::Oidn,
        spp: 8,
        ..Default::default()
    };
    r.render(&settings, true);
    r.set_worlds(&[vec![], vec![]]).unwrap();
    r.render(&settings, true);
    let raw = r.read_pixels().unwrap();
    let generation = r.generation;
    r.finish_denoising(&settings).unwrap();
    assert_eq!(r.generation, generation);
    assert!(r.oidn.as_ref().unwrap().ready(r.samples));
    assert_eq!(
        bytemuck::cast_slice::<Pixel, u8>(&raw),
        bytemuck::cast_slice::<Pixel, u8>(&r.read_pixels().unwrap())
    );
    let a = std::env::temp_dir().join(format!("oidn-clean-{}.png", std::process::id()));
    let b = a.with_extension("raw.png");
    r.save_png(&a).unwrap();
    r.save_raw_png(&b).unwrap();
    let read = |path: &std::path::Path| {
        let decoder =
            png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
        let mut reader = decoder.read_info().unwrap();
        let mut data = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut data).unwrap();
        data.truncate(info.buffer_size());
        data
    };
    let clean = read(&a);
    let original = read(&b);
    let output = r.oidn.as_ref().unwrap().output_buffer();
    let staging = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = r.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(output, 0, &staging, 0, output.size());
    let bytes = r.read_buffer(encoder, &staging).unwrap();
    for pixel in bytes
        .as_chunks::<{ size_of::<super::oidn::Feedback>() }>()
        .0
    {
        let pixel: super::oidn::Feedback = bytemuck::pod_read_unaligned(pixel);
        for (actual, expected) in pixel.raw_a[..3].iter().zip([0.045, 0.055, 0.075]) {
            assert!(
                (*actual - expected).abs() < 1e-6,
                "stale scene input reached the output"
            );
        }
    }
    let mean_difference = clean
        .iter()
        .zip(&original)
        .map(|(a, b)| a.abs_diff(*b) as f32)
        .sum::<f32>()
        / clean.len() as f32;
    eprintln!(
        "Constant-sky PNG: mean difference {mean_difference}, max {}",
        clean
            .iter()
            .zip(&original)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap()
    );
    assert!(
        mean_difference < 2.,
        "a stale image replaced the empty scene"
    );
    std::fs::remove_file(a).unwrap();
    std::fs::remove_file(b).unwrap();
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn adaptive_gpu_queue_keeps_exact_budget_and_constant_radiance() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 33, 25, 42).unwrap();
    r.set_worlds(&[vec![], vec![]]).unwrap();
    let s = Settings {
        mode: Mode::WorldA,
        sampling: crate::settings::SamplingMode::Variance,
        spp: 1,
        ..Default::default()
    };
    for _ in 0..16 {
        r.accumulate(&s);
    }
    let mut pixels = r.read_pixels().unwrap();
    assert!(pixels.iter().all(|p| p.counts[0] == 16));
    for (i, p) in pixels.iter_mut().enumerate() {
        // A controlled variance signal isolates scheduling from scene geometry.
        p.moment_a[3] = if i % 33 < 16 { 1600. } else { 0. };
        p.stats_a[3] = 16.;
    }
    r.queue
        .write_buffer(&r.pixels, 0, bytemuck::cast_slice(&pixels));
    for _ in 0..64 {
        r.accumulate(&s);
    }
    let pixels = r.read_pixels().unwrap();
    let n = pixels.len();
    assert_eq!(
        pixels.iter().map(|p| p.counts[0] as u64).sum::<u64>(),
        n as u64 * r.samples as u64
    );
    assert_eq!(pixels.iter().map(|p| p.schedule[0]).sum::<u32>(), n as u32);
    let left = pixels
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 33 < 16)
        .map(|(_, p)| p.counts[0] as f64 - 16.)
        .sum::<f64>()
        / (16. * 25.);
    let right = pixels
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 33 >= 16)
        .map(|(_, p)| p.counts[0] as f64 - 16.)
        .sum::<f64>()
        / (17. * 25.);
    eprintln!("Adaptive rays after warmup: high variance {left}, clean {right}");
    assert!(left > right * 2.0);
    let queued = pixels[0].allocation[0] as usize;
    assert_eq!(queued, pixels.iter().filter(|p| p.schedule[0] > 0).count());
    let mut unique = std::collections::HashSet::new();
    for task in pixels.iter().take(queued) {
        let id = task.work[0] as usize;
        assert!(id < n && pixels[id].schedule[0] > 0);
        assert!(unique.insert(id));
    }
    for p in &pixels {
        assert!(p.schedule[0] <= 6);
        assert_eq!(p.counts[2], 0, "corrupt queue guard fired");
        assert_eq!(
            p.light_a[3], r.samples as f32,
            "photon epochs must not depend on camera allocation"
        );
        for (value, expected) in p.a[..3].iter().zip([0.045, 0.055, 0.075]) {
            assert!((*value / r.samples as f32 - expected).abs() < 2e-6);
        }
    }
    // Paused resets and resizing cannot dispatch a stale queue.
    r.reset();
    r.render(&s, false);
    assert_eq!(r.samples, 0);
    assert!(r.read_pixels().unwrap().iter().all(|p| p.counts[0] == 0));
    r.resize(65, 17).unwrap();
    r.accumulate(&s);
    assert!(r.read_pixels().unwrap().iter().all(|p| p.counts[0] == 1));
    // More than 256 scan blocks exercises carries between global scan chunks.
    r.resize(513, 387).unwrap();
    for _ in 0..20 {
        r.accumulate(&s);
    }
    let large = r.read_pixels().unwrap();
    assert_eq!(
        large.iter().map(|p| u64::from(p.counts[0])).sum::<u64>(),
        u64::from(r.width * r.height * r.samples)
    );
    assert_eq!(
        large.iter().map(|p| p.schedule[0]).sum::<u32>(),
        r.width * r.height
    );
    assert!(large.iter().all(|p| p.counts[2] == 0));
}

#[test]
#[ignore = "requires installed OIDN runtime and Metal GPU"]
fn denoising_aware_allocation_publishes_reconstruction_error_and_preserves_budgets() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 48, 32, 42).unwrap();
    let s = Settings {
        mode: Mode::WorldA,
        worlds: [
            scene::WorldPreset::GlassGallery,
            scene::WorldPreset::CrystalGarden,
        ],
        caustics: true,
        denoise: true,
        denoiser: crate::settings::DenoiserKind::Oidn,
        sampling: crate::settings::SamplingMode::Denoised,
        depth: 20,
        spp: 4,
        ..Default::default()
    };
    for _ in 0..12 {
        r.accumulate(&s);
    }
    let before = r.read_pixels().unwrap();
    r.finish_denoising(&s).unwrap();
    let after = r.read_pixels().unwrap();
    for (a, b) in before.iter().zip(&after) {
        assert_eq!(a.a, b.a);
        assert_eq!(a.counts, b.counts);
    }
    let priors: Vec<_> = after.iter().map(|p| p.guide_normal_a[3]).collect();
    assert!(priors.iter().all(|v| v.is_finite() && *v > 0.));
    assert!(
        priors.iter().any(|v| *v > 1e-6),
        "no denoising uncertainty reached the sampler"
    );
    assert_eq!(
        after.iter().map(|p| p.counts[0] as u64).sum::<u64>(),
        u64::from(r.width * r.height * r.samples)
    );
    for _ in 0..8 {
        r.accumulate(&s);
    }
    let after = r.read_pixels().unwrap();
    assert_eq!(
        after.iter().map(|p| p.counts[0] as u64).sum::<u64>(),
        u64::from(r.width * r.height * r.samples)
    );
    assert!(after.iter().any(|p| p.counts[0] < r.samples));
    assert!(after.iter().any(|p| p.counts[0] > r.samples));
    assert!(after.iter().flat_map(|p| p.a).all(f32::is_finite));
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn adaptive_control_variates_preserve_analytic_world_mixture_and_photon_energy() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 33, 25, 42).unwrap();
    let floor = Object {
        center_radius: [0.; 4],
        extent_shape: [0., 0., 0., 2.],
        color_material: [0.; 4],
        emission: [1., 1., 1., 0.],
    };
    let brighter = Object {
        emission: [4., 4., 4., 0.],
        ..floor
    };
    r.set_worlds(&[vec![floor], vec![brighter]]).unwrap();
    let mut s = Settings {
        mode: Mode::PerPath,
        p: 0.2,
        sampling: crate::settings::SamplingMode::Variance,
        spp: 8,
        pitch: 0.7,
        ..Default::default()
    };
    for _ in 0..64 {
        r.accumulate(&s);
    }
    let pixels = r.read_pixels().unwrap();
    let mean = pixels
        .iter()
        .map(|p| p.a[0] as f64 / r.samples as f64)
        .sum::<f64>()
        / pixels.len() as f64;
    eprintln!("Adaptive analytic mixture: {mean}, expected 3.4");
    assert!((mean - 3.4).abs() < 0.012);
    assert_eq!(
        pixels.iter().map(|p| u64::from(p.counts[0])).sum::<u64>(),
        u64::from(r.width * r.height * r.samples)
    );
    let receiver = Object {
        color_material: [0.75, 0.75, 0.75, 0.],
        emission: [0.; 4],
        ..floor
    };
    let lens = Object {
        center_radius: [0., 1.6, 0., 0.9],
        extent_shape: [0.; 4],
        color_material: [0.9, 0.95, 1., 3.5],
        emission: [0.; 4],
    };
    let lamp = Object {
        center_radius: [0., 6.5, 0., 0.3],
        extent_shape: [0.; 4],
        color_material: [0.; 4],
        emission: [100., 100., 100., 0.],
    };
    r.set_worlds(&[vec![receiver, lens, lamp], vec![]]).unwrap();
    s.mode = Mode::WorldA;
    s.caustics = true;
    s.distance = 8.;
    s.yaw = 0.;
    s.depth = 12;
    let mut estimates = Vec::new();
    for mode in [
        crate::settings::SamplingMode::Uniform,
        crate::settings::SamplingMode::Variance,
    ] {
        s.sampling = mode;
        r.reset();
        for _ in 0..16 {
            r.accumulate(&s);
        }
        estimates.push(r.read_pixels().unwrap());
    }
    let mut a = 0_f64;
    let mut b = 0_f64;
    for (first, second) in estimates[0].iter().zip(&estimates[1]) {
        for c in 0..3 {
            assert!(
                (first.light_a[c] - second.light_a[c]).abs() < 1e-4 * (1. + first.light_a[c].abs()),
                "visible caustics were normalized by adaptive camera counts"
            );
            a += first.a[c] as f64;
            b += second.a[c] as f64;
        }
    }
    eprintln!("Adaptive caustic total energy ratio {}", b / a);
    assert!((b / a - 1.).abs() < 0.05);
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn adaptive_world_modes_neural_cache_and_legacy_preview_remain_finite() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 24, 16, 42).unwrap();
    let mut s = Settings {
        sampling: crate::settings::SamplingMode::Variance,
        neural_cache: true,
        spp: 4,
        ..Default::default()
    };
    for mode in Mode::ALL {
        s.mode = mode;
        r.reset();
        for _ in 0..8 {
            r.accumulate(&s);
        }
        assert_eq!(r.neural_training_batches(), 32);
        let pixels = r.read_pixels().unwrap();
        assert_eq!(
            pixels.iter().map(|p| u64::from(p.counts[0])).sum::<u64>(),
            24 * 16 * 32
        );
        if mode == Mode::Blend {
            assert_eq!(
                pixels.iter().map(|p| u64::from(p.counts[1])).sum::<u64>(),
                24 * 16 * 32
            );
        }
        assert!(
            pixels
                .iter()
                .flat_map(|p| p.a.into_iter().chain(p.b))
                .all(f32::is_finite)
        );
    }
    s.neural_cache = false;
    s.mode = Mode::Blend;
    s.denoise = true;
    s.denoiser = crate::settings::DenoiserKind::Temporal;
    r.reset();
    for _ in 0..5 {
        r.render(&s, true);
    }
    assert!(
        r.read_linear_image(&s, true)
            .unwrap()
            .iter()
            .flatten()
            .all(|v| v.is_finite() && *v >= 0.)
    );
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn adaptive_gpu_scheduler_uses_denoised_error_not_only_raw_variance() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 33, 25, 42).unwrap();
    r.set_worlds(&[vec![], vec![]]).unwrap();
    let mut pixels = vec![Pixel::zeroed(); 33 * 25];
    for (i, p) in pixels.iter_mut().enumerate() {
        p.a = [0.72, 0.88, 1.2, 16.];
        p.moment_a = [0.045, 0.055, 0.075, 0.];
        p.counts[0] = 16;
        p.stats_a[3] = 16.;
        p.guide_normal_a[3] = if i % 33 < 16 { 100. } else { 1e-20 };
    }
    let allocation = super::adaptive::Adaptive::new(&r.device, &r.shader, &r.pipeline_layout);
    let mut ratios = Vec::new();
    for mode in [1., 2.] {
        let mut params = Params::zeroed();
        params.size = [33, 25, 1, 16];
        params.transport = [0, 4, 42, 0];
        params.reconstruction = [1., mode, 16., 0.];
        r.queue
            .write_buffer(&r.uniform, 0, bytemuck::bytes_of(&params));
        r.queue
            .write_buffer(&r.pixels, 0, bytemuck::cast_slice(&pixels));
        let mut encoder = r.device.create_command_encoder(&Default::default());
        allocation.prepare(
            &mut encoder,
            &r.bind_group,
            &r.pixels,
            [33, 25],
            [16, 1],
            None,
        );
        r.queue.submit([encoder.finish()]);
        let result = r.read_pixels().unwrap();
        assert_eq!(result.iter().map(|p| p.schedule[0]).sum::<u32>(), 33 * 25);
        let left = result
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 33 < 16)
            .map(|(_, p)| p.schedule[0] as f32)
            .sum::<f32>()
            / (16. * 25.);
        let right = result
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 33 >= 16)
            .map(|(_, p)| p.schedule[0] as f32)
            .sum::<f32>()
            / (17. * 25.);
        ratios.push(left / right);
    }
    assert!(
        (ratios[0] - 1.).abs() < 0.01,
        "variance-only sampling should ignore the injected denoiser prior"
    );
    assert!(
        ratios[1] > 3.,
        "denoised-error sampling did not use reconstruction uncertainty: {ratios:?}"
    );
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn advanced_caustic_proposals_preserve_energy_and_reset_reuse() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 32, 24, 42).unwrap();
    let floor = Object {
        center_radius: [0.; 4],
        extent_shape: [0., 0., 0., 2.],
        color_material: [0.75, 0.75, 0.75, 0.],
        emission: [0.; 4],
    };
    let sphere = Object {
        center_radius: [0., 1.6, 0., 0.9],
        extent_shape: [0.; 4],
        color_material: [0.85, 0.95, 0.99, 3.5],
        emission: [0.; 4],
    };
    let light = Object {
        center_radius: [0., 6.5, 0., 0.3],
        extent_shape: [0.; 4],
        color_material: [0.; 4],
        emission: [100., 100., 100., 0.],
    };
    r.set_worlds(&[vec![floor, sphere, light], vec![]]).unwrap();
    let mut settings = Settings {
        mode: Mode::WorldA,
        caustics: true,
        photon_budget: 4096,
        depth: 12,
        yaw: 0.,
        pitch: 0.7,
        distance: 8.,
        ..Default::default()
    };
    let diagnostics = |r: &Renderer| -> [u32; 8] {
        let c = r.caustics.as_ref().unwrap();
        let stage = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Caustic method diagnostics"),
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = r.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&c.data, c.diagnostic_offset, &stage, 0, 32);
        bytemuck::pod_read_unaligned(&r.read_buffer(encoder, &stage).unwrap())
    };
    let mut baseline = 0.;
    for (guided, reuse, manifold) in [
        (false, false, false),
        (true, false, false),
        (false, true, false),
        (false, false, true),
        (true, true, true),
    ] {
        settings.photon_guiding = guided;
        settings.photon_reuse = reuse;
        settings.manifold_guiding = manifold;
        r.reset();
        for _ in 0..128 {
            r.accumulate(&settings);
            r.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(60)),
                })
                .unwrap();
        }
        let pixels = r.read_pixels().unwrap();
        let energy = pixels
            .iter()
            .map(|p| p.light_a[..3].iter().sum::<f32>() as f64)
            .sum::<f64>()
            / (pixels.len() * 3 * r.samples as usize) as f64;
        let d = diagnostics(&r);
        eprintln!(
            "Caustic methods guided={guided},reuse={reuse},manifold={manifold}: visible energy {energy}, diagnostics {d:?}"
        );
        assert!(
            pixels
                .iter()
                .all(|p| p.a.iter().chain(&p.light_a).all(|v| v.is_finite()))
        );
        if !guided && !reuse && !manifold {
            baseline = energy;
            assert!(baseline > 0.001);
        } else {
            assert!(
                (energy / baseline - 1.).abs() < 0.2,
                "energy ratio {}",
                energy / baseline
            );
        }
        if manifold {
            assert!(
                d[0] > 0 && d[1] > 0,
                "Newton transmission proposals must actually converge"
            );
        }
        if reuse {
            assert!(d[2] > 0, "reservoir sources must actually be reused");
        }
    }
    // A reset invalidates both reservoir banks and the learned manifold focus.
    r.reset();
    r.accumulate(&settings);
    let reset = diagnostics(&r);
    assert_eq!(reset[0], 0);
    assert_eq!(reset[2], 0);
    assert_eq!(r.samples, 1);
    // A non-power-of-two launch count and a different budget remain normalized.
    settings.photon_budget = 777;
    r.accumulate(&settings);
    assert_eq!(
        r.samples, 1,
        "changing light budget restarts both estimators"
    );
    assert_eq!(diagnostics(&r)[2], 0);
    // Two caster proposals must learn distinct probabilities from real
    // contributions; a feature switch that leaves every proposal uniform fails.
    let mut far = sphere;
    far.center_radius[0] = 6.;
    r.set_worlds(&[vec![floor, sphere, far, light], vec![]])
        .unwrap();
    settings.photon_guiding = true;
    settings.photon_reuse = false;
    settings.manifold_guiding = false;
    settings.photon_budget = 2048;
    for _ in 0..48 {
        r.accumulate(&settings);
    }
    let c = r.caustics.as_ref().unwrap();
    let stage = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Learned caster probabilities"),
        size: 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = r.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&c.data, c.diagnostic_offset + 32 + 1024 * 4, &stage, 0, 8);
    let probability: [f32; 2] =
        bytemuck::pod_read_unaligned(&r.read_buffer(encoder, &stage).unwrap());
    eprintln!("Learned caster probabilities {probability:?}");
    assert!((probability.iter().sum::<f32>() - 1.).abs() < 1e-5);
    assert!((probability[0] - 0.5).abs() > 0.03);
    assert!(probability.iter().all(|p| *p >= 0.099));
    // Beyond the bounded guide table, both light AND caster PDFs must use the
    // baseline; a zero cached caster PDF would brighten the cone proposal.
    let mut many = vec![floor, sphere];
    for i in 0..33 {
        let mut l = light;
        l.center_radius[0] = (i as f32 - 16.) * 0.7;
        l.center_radius[3] = 0.15;
        many.push(l);
    }
    r.set_worlds(&[many, vec![]]).unwrap();
    let mut images = Vec::new();
    for guided in [false, true] {
        settings.photon_guiding = guided;
        r.reset();
        for _ in 0..16 {
            r.accumulate(&settings);
        }
        images.push(r.read_pixels().unwrap());
    }
    let difference = images[0]
        .iter()
        .zip(&images[1])
        .flat_map(|(a, b)| (0..3).map(move |c| (a.a[c] - b.a[c]).abs()))
        .fold(0_f32, f32::max);
    assert!(
        difference < 0.0001,
        "bounded-table fallback changed radiance: {difference}"
    );
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn oidn_live_display_holds_export_quality_while_new_raw_samples_arrive() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let r = Renderer::new(device, queue, 4, 4, 42).unwrap();
    let denoiser = oidn::Oidn::new(&r.device, &r.shader, &r.layout, [4, 4]);
    let snapshot = oidn::Feedback {
        color_a: [0.2, 0.5, 0.8, 8.],
        raw_a: [0.1, 0.2, 0.4, -1.],
        color_b: [0.8, 0.4, 0.1, 8.],
        raw_b: [0.5, 0.3, 0.1, -1.],
    };
    r.queue.write_buffer(
        denoiser.output_buffer(),
        0,
        bytemuck::cast_slice(&[snapshot; 16]),
    );
    let stage = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Live OIDN display regression"),
        size: 256 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let show = |params: &Params, values: &[Pixel]| {
        r.queue
            .write_buffer(&r.uniform, 0, bytemuck::bytes_of(params));
        r.queue
            .write_buffer(&r.pixels, 0, bytemuck::cast_slice(values));
        let mut encoder = r.device.create_command_encoder(&Default::default());
        denoiser.present(&mut encoder, &r.bind_group);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &r.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &stage,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(4),
                },
            },
            r.texture.size(),
        );
        let bytes = r.read_buffer(encoder, &stage).unwrap();
        bytes
            .as_chunks::<256>()
            .0
            .iter()
            .flat_map(|row| row[..16].iter().copied())
            .collect::<Vec<_>>()
    };
    for mode in [Mode::WorldA, Mode::WorldB, Mode::Blend] {
        for strength in [0., 0.8, 1.] {
            let mut params = Params::zeroed();
            params.size = [4, 4, 0, 8];
            params.transport = [mode as u32, 20, 42, View::Radiance as u32];
            params.controls = [0.35, 0.8, 0., 0.];
            params.reconstruction[0] = strength;
            let mut original = [Pixel::zeroed(); 16];
            for p in &mut original {
                for c in 0..3 {
                    p.a[c] = snapshot.raw_a[c] * 8.;
                    p.b[c] = snapshot.raw_b[c] * 8.;
                }
            }
            let finished_export = show(&params, &original);
            // Eight newer, deliberately noisy samples arrive during OIDN work.
            // Both raw sums and the global sample count change; feedback does not.
            let mut current = original;
            for (i, p) in current.iter_mut().enumerate() {
                for c in 0..3 {
                    p.a[c] += if (i + c) % 2 == 0 { 12. } else { 0. };
                    p.b[c] += if (i + c) % 2 == 0 { 0. } else { 20. };
                }
            }
            params.size[3] = 16;
            let live = show(&params, &current);
            assert_eq!(
                live, finished_export,
                "{mode:?}, strength {strength}: raw samples leaked into the reconstructed display"
            );
            assert!(finished_export.iter().step_by(4).any(|v| *v > 0));
            // Exposure remains responsive without replacing the completed image.
            params.controls[2] = 1.;
            assert_ne!(show(&params, &current), live);
        }
    }
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn interactive_glass_work_stays_bounded_when_ui_outpaces_gpu() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 800, 600, 42).unwrap();
    let settings = Settings {
        mode: Mode::WorldA,
        worlds: [
            scene::WorldPreset::GlassGallery,
            scene::WorldPreset::CrystalGarden,
        ],
        caustics: true,
        photon_guiding: true,
        denoise: true,
        depth: 20,
        yaw: 0.6,
        pitch: 0.55,
        distance: 13.,
        ..Default::default()
    };
    let completed = Arc::new(AtomicUsize::new(0));
    let mut submitted = 0;
    let mut worst_backlog = 0;
    let mut first_submit = Duration::ZERO;
    for frame in 0..24 {
        let started = std::time::Instant::now();
        if r.render_live(&settings, true) {
            submitted += 1;
            let completed = completed.clone();
            r.queue.on_submitted_work_done(move || {
                completed.fetch_add(1, Ordering::Release);
            });
        }
        if frame == 0 {
            first_submit = started.elapsed();
        }
        r.device.poll(wgpu::PollType::Poll).unwrap();
        worst_backlog = worst_backlog.max(submitted - completed.load(Ordering::Acquire));
    }
    let wait = std::time::Instant::now();
    r.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(60)),
        })
        .unwrap();
    eprintln!(
        "Live glass: first submit {first_submit:?}, submissions {submitted}, maximum GPU backlog {worst_backlog}, queued drain {:?}",
        wait.elapsed()
    );
    assert!(
        worst_backlog <= 1,
        "UI input accumulated behind {worst_backlog} full glass frames"
    );
    assert_eq!(r.samples as usize, submitted);
    // Once completed, a new render and pending settings changes can progress.
    assert!(r.render_live(&settings, true));
}

#[test]
#[ignore = "requires installed OIDN runtime and Metal GPU"]
fn asynchronous_image_exports_match_sync_without_resetting_accumulation() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 32, 24, 42).unwrap();
    let settings = Settings {
        mode: Mode::WorldA,
        denoise: true,
        spp: 8,
        ..Default::default()
    };
    r.set_worlds(&[vec![], vec![]]).unwrap();
    r.render(&settings, true);
    r.finish_denoising(&settings).unwrap();
    let raw = r.read_pixels().unwrap();
    let count = r.samples;
    let dir = std::env::temp_dir().join(format!("multiworld-async-export-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    r.save_png(&dir.join("sync.png")).unwrap();
    r.save_raw_png(&dir.join("sync-raw.png")).unwrap();
    let filtered = r.start_image_export(&dir.join("async.png"), false).unwrap();
    let unfiltered = r
        .start_image_export(&dir.join("async-raw.png"), true)
        .unwrap();
    assert_eq!(r.samples, count);
    assert_eq!(
        bytemuck::cast_slice::<Pixel, u8>(&raw),
        bytemuck::cast_slice::<Pixel, u8>(&r.read_pixels().unwrap())
    );
    // Camera work can continue after the copied snapshot is queued; PNG
    // encoding owns its data and must not accidentally export the newer image.
    r.accumulate(&settings);
    assert_eq!(r.samples, count + 8);
    let wait = |job: super::export::ImageExport| {
        let started = std::time::Instant::now();
        loop {
            if let Some(result) = job.poll() {
                result.unwrap();
                break;
            }
            assert!(started.elapsed() < Duration::from_secs(30));
            r.device.poll(wgpu::PollType::Poll).unwrap();
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    wait(filtered);
    wait(unfiltered);
    for (a, b) in [("sync.png", "async.png"), ("sync-raw.png", "async-raw.png")] {
        assert_eq!(
            std::fs::read(dir.join(a)).unwrap(),
            std::fs::read(dir.join(b)).unwrap()
        );
    }
    let invalid = r.start_image_export(&dir, false).unwrap();
    let started = std::time::Instant::now();
    loop {
        if let Some(result) = invalid.poll() {
            assert!(result.is_err());
            break;
        }
        assert!(started.elapsed() < Duration::from_secs(30));
        r.device.poll(wgpu::PollType::Poll).unwrap();
        std::thread::sleep(Duration::from_millis(1));
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn photon_nrc_sppc_targets_match_direct_lighting_and_keep_query_lifetimes() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 19, 13, 42).unwrap();
    let floor = Object {
        center_radius: [0.; 4],
        extent_shape: [0., 0., 0., 2.],
        color_material: [0.8, 0.8, 0.8, 0.],
        emission: [0.; 4],
    };
    let light = Object {
        center_radius: [0., 4., 0., 0.7],
        extent_shape: [0.; 4],
        color_material: [0.; 4],
        emission: [10., 10., 10., 0.],
    };
    r.set_worlds(&[vec![floor, light], vec![]]).unwrap();
    let s = Settings {
        mode: Mode::WorldA,
        photon_nrc: true,
        photon_nrc_queries: 1024,
        photon_nrc_hash_log2: 10,
        photon_nrc_photons: 8192,
        photon_nrc_global_radius: 0.5,
        photon_nrc_caustic_radius: 0.2,
        depth: 8,
        yaw: 0.,
        pitch: 0.9,
        distance: 6.,
        ..Default::default()
    };
    for _ in 0..80 {
        r.accumulate(&s);
        r.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(60)),
            })
            .unwrap();
    }
    let cache = r.photon_nrc.as_ref().unwrap();
    let read = |offset: u64, len: u64| {
        let stage = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SPPC validation"),
            size: len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = r.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&cache.data, offset, &stage, 0, len);
        r.read_buffer(encoder, &stage).unwrap()
    };
    let state: [u32; 4] = bytemuck::pod_read_unaligned(&read(0, 16));
    assert_eq!(state[0], 80 * 8192);
    assert!(state[2] >= 32);
    let queries = read(cache.query_offset(), 1024 * 128);
    let mut actual = 0_f64;
    let mut expected = 0_f64;
    let mut matched = 0;
    let mut shrunk = 0;
    for bytes in queries.as_chunks::<128>().0 {
        let q: [[f32; 4]; 8] = bytemuck::pod_read_unaligned(bytes);
        if q[6][3].to_bits() == 0 {
            continue;
        }
        let age = state[0].wrapping_sub(q[6][2].to_bits());
        assert!(age > 0 && age <= 51 * 8192, "incorrect ring lifetime {age}");
        if q[6][0] < 0.25 {
            shrunk += 1;
        }
        let p = Vec3::from_array(q[0][..3].try_into().unwrap());
        if p.y.abs() > 0.001 || p.x * p.x + p.z * p.z > 9. {
            continue;
        }
        let d2 = p.x * p.x + p.z * p.z + 16.;
        let factor = 0.49 * 4. / d2.powf(1.5);
        let target = 0.8 * (10. * factor + 0.045 * (1. - factor));
        let value = (q[4][0] / q[6][0] + q[5][0] / q[6][1]) / (std::f32::consts::PI * age as f32)
            + q[7][0] / q[7][3].max(1.);
        assert!(value.is_finite() && value >= 0.);
        actual += value as f64;
        expected += target as f64;
        matched += 1;
    }
    eprintln!(
        "SPPC: state {state:?}, {matched} analytic queries, target ratio {}, {shrunk} shrinking global radii",
        actual / expected
    );
    assert!(matched > 20 && shrunk > 0);
    assert!(
        (actual / expected - 1.).abs() < 0.2,
        "photon target normalization ratio {}",
        actual / expected
    );
    assert!(
        r.read_pixels()
            .unwrap()
            .iter()
            .all(|p| p.a.iter().all(|v| v.is_finite()))
    );
    let hash_offset = 4_495_936;
    let weights = read(hash_offset, 32 * 1024 * 4);
    let largest = weights
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b).abs())
        .fold(0_f32, f32::max);
    assert!(largest > 0.001, "hash features were not trained: {largest}");
    cache.verify_matrix_predictions(&r);
    let batches = r.photon_nrc_batches();
    r.clear_accumulation();
    r.accumulate(&s);
    assert_eq!(
        r.photon_nrc_batches(),
        batches + 1,
        "camera/image reset must retain query learning"
    );
    r.reset();
    r.accumulate(&s);
    assert_eq!(r.photon_nrc_batches(), 1);
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn photon_nrc_glass_caustics_and_blend_worlds_are_isolated() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 16, 12, 42).unwrap();
    let floor = Object {
        center_radius: [0.; 4],
        extent_shape: [0., 0., 0., 2.],
        color_material: [0.8, 0.8, 0.8, 0.],
        emission: [0.; 4],
    };
    let glass = Object {
        center_radius: [0., 1.5, 0., 0.7],
        extent_shape: [0.; 4],
        color_material: [1., 1., 1., 3.5],
        emission: [0.; 4],
    };
    let light = Object {
        center_radius: [0., 4., 0., 0.7],
        extent_shape: [0.; 4],
        color_material: [0.; 4],
        emission: [10., 10., 10., 0.],
    };
    r.set_worlds(&[vec![floor, glass, light], vec![floor, glass]])
        .unwrap();
    let s = Settings {
        mode: Mode::Blend,
        photon_nrc: true,
        photon_nrc_queries: 1024,
        photon_nrc_hash_log2: 10,
        photon_nrc_photons: 8192,
        photon_nrc_global_radius: 0.5,
        photon_nrc_caustic_radius: 0.25,
        depth: 12,
        yaw: 0.,
        pitch: 0.9,
        distance: 6.,
        ..Default::default()
    };
    for _ in 0..64 {
        r.accumulate(&s);
        r.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(60)),
            })
            .unwrap();
    }
    let cache = r.photon_nrc.as_ref().unwrap();
    let stage = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SPPC world isolation"),
        size: 1024 * 128,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut e = r.device.create_command_encoder(&Default::default());
    e.copy_buffer_to_buffer(&cache.data, cache.query_offset(), &stage, 0, 1024 * 128);
    let bytes = r.read_buffer(e, &stage).unwrap();
    let mut caustic = 0_f64;
    let mut counts = [0; 2];
    for bytes in bytes.as_chunks::<128>().0 {
        let q: [[f32; 4]; 8] = bytemuck::pod_read_unaligned(bytes);
        if q[6][3].to_bits() == 0 {
            continue;
        }
        let world = q[0][3] as usize;
        counts[world] += 1;
        if world == 0 {
            caustic += q[5][0] as f64;
        } else {
            assert_eq!(q[4], [0.; 4]);
            assert_eq!(q[5], [0.; 4]);
        }
    }
    assert!(counts.iter().all(|n| *n > 50), "{counts:?}");
    assert!(caustic > 0., "glass never trained caustic queries");
    let pixels = r.read_pixels().unwrap();
    assert!(
        pixels
            .iter()
            .all(|p| p.a.iter().chain(&p.b).all(|x| x.is_finite()))
    );
    assert!(
        pixels.iter().any(|p| p.b[0] > 0.),
        "environment in unlit world B disappeared"
    );
}

#[test]
#[ignore = "requires Metal compute adapter"]
fn photon_nrc_camera_orbit_preserves_ground_shadow_contrast() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 64, 48, 42).unwrap();
    let floor = Object {
        center_radius: [0.; 4],
        extent_shape: [0., 0., 0., 2.],
        color_material: [0.8, 0.8, 0.8, 0.],
        emission: [0.; 4],
    };
    let block = Object {
        center_radius: [0., 0.8, 0., 0.],
        extent_shape: [0.5, 0.8, 0.5, 1.],
        color_material: [0., 0., 0., 0.],
        emission: [0.; 4],
    };
    let light = Object {
        center_radius: [-1.5, 4., 0., 0.3],
        extent_shape: [0.; 4],
        color_material: [0.; 4],
        emission: [100., 100., 100., 0.],
    };
    let glass = Object {
        center_radius: [1.5, 0.7, 0., 0.6],
        extent_shape: [0.; 4],
        color_material: [1., 1., 1., 3.5],
        emission: [0.; 4],
    };
    r.set_worlds(&[vec![floor, block, light, glass], vec![]])
        .unwrap();
    let mut s = Settings {
        mode: Mode::WorldA,
        photon_nrc: true,
        photon_nrc_queries: 2048,
        photon_nrc_hash_log2: 14,
        photon_nrc_photons: 32768,
        depth: 12,
        yaw: -0.8,
        pitch: 0.9,
        distance: 6.,
        ..Default::default()
    };
    let wait = |r: &Renderer| {
        r.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(60)),
            })
            .unwrap();
    };
    for _ in 0..40 {
        r.accumulate(&s);
        wait(&r);
    }
    s.yaw = 1.2;
    r.accumulate(&s);
    wait(&r);
    let immediate = r.read_pixels().unwrap();
    for _ in 0..63 {
        r.accumulate(&s);
        wait(&r);
    }
    let settled = r.read_pixels().unwrap();
    s.photon_nrc = false;
    s.spp = 8;
    for _ in 0..32 {
        r.accumulate(&s);
        wait(&r);
    }
    let reference = r.read_pixels().unwrap();
    for (stage, cached) in [
        ("first orbit image", immediate),
        ("corrected cache image", settled),
    ] {
        let mut shadow = [0_f64; 3];
        let mut lit = [0_f64; 3];
        for (c, p) in cached.iter().zip(&reference) {
            if p.normals_a[1] / p.a[3] < 0.99 {
                continue;
            }
            let actual = (c.a[0] / c.a[3]) as f64;
            let expected = (p.a[0] / p.a[3]) as f64;
            let target = if expected < 0.075 {
                &mut shadow
            } else if expected > 0.3 && expected < 0.8 {
                &mut lit
            } else {
                continue;
            };
            target[0] += actual;
            target[1] += expected;
            target[2] += 1.;
        }
        assert!(
            shadow[2] > 10. && lit[2] > 10.,
            "missing ground regions: {shadow:?} {lit:?}"
        );
        let ratio =
            (lit[0] / lit[2] - shadow[0] / shadow[2]) / (lit[1] / lit[2] - shadow[1] / shadow[2]);
        eprintln!("{stage}: orbit shadow contrast ratio {ratio}, shadow {shadow:?}, lit {lit:?}");
        assert!(
            ratio > 0.7,
            "{stage}: orbit erased ground shadow contrast: {ratio}"
        );
    }
}

#[test]
#[ignore = "requires Metal compute adapter"]
fn photon_nrc_corrected_caustic_energy_survives_camera_orbit() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 32, 24, 42).unwrap();
    let floor = Object {
        center_radius: [0.; 4],
        extent_shape: [0., 0., 0., 2.],
        color_material: [0.75, 0.75, 0.75, 0.],
        emission: [0.; 4],
    };
    let lens = Object {
        center_radius: [0., 1.6, 0., 0.9],
        extent_shape: [0.; 4],
        color_material: [1., 1., 1., 3.5],
        emission: [0.; 4],
    };
    let light = Object {
        center_radius: [0., 6.5, 0., 0.3],
        extent_shape: [0.; 4],
        color_material: [0.; 4],
        emission: [100., 100., 100., 0.],
    };
    r.set_worlds(&[vec![floor, lens, light], vec![]]).unwrap();
    let mut s = Settings {
        mode: Mode::WorldA,
        photon_nrc: true,
        photon_nrc_queries: 1024,
        photon_nrc_hash_log2: 10,
        photon_nrc_photons: 8192,
        depth: 12,
        yaw: -0.8,
        pitch: 0.9,
        distance: 6.,
        ..Default::default()
    };
    let wait = |r: &Renderer| {
        r.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(60)),
            })
            .unwrap();
    };
    for _ in 0..40 {
        r.accumulate(&s);
        wait(&r);
    }
    s.yaw = 1.2;
    for _ in 0..16 {
        r.accumulate(&s);
        wait(&r);
    }
    let [reference, corrected] = r.photon_nrc.as_ref().unwrap().caustic_probe_means(&r);
    eprintln!("After orbit, ground caustic: reference {reference}, corrected cache {corrected}");
    assert!(reference > 2., "fixture did not focus light");
    assert!(
        (corrected / reference - 1.).abs() < 0.15,
        "cache lost focused caustic energy"
    );
}

#[test]
#[ignore = "requires hardware ray-query adapter"]
fn hardware_ray_query_single_sphere() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 8, 8, 42).unwrap();
    assert!(r.hardware_ray_tracing_available());
    r.set_worlds(&[vec![object([0.; 3], 0., [1., 0., 0.])], vec![]])
        .unwrap();
    let rays = [probe(Vec3::new(0., 0., -3.), Vec3::Z, 0)];
    assert_queries(&queries_with_policy(&r, &rays, 48));
}

#[test]
#[ignore = "requires hardware ray-query adapter"]
fn hardware_ray_query_build_order_preserves_translated_emitters() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 8, 8, 42).unwrap();
    let scenes = scene::worlds(42);
    let rays = [probe(
        Vec3::new(-10.618146, 4.6591196, -2.3284006),
        Vec3::new(0.9945289, -0.0023215641, 0.10443557),
        0,
    )];
    for scene in [
        vec![scenes[0][26]],
        vec![scenes[0][0], scenes[0][26]],
        scenes[0][1..].to_vec(),
        scenes[0].clone(),
    ] {
        r.set_worlds(&[scene, vec![]]).unwrap();
        assert_queries(&queries_with_policy(&r, &rays, 48));
    }
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn optional_acceleration_features_have_a_real_software_fallback() {
    let instance = wgpu::Instance::new(instance_descriptor());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Portable fallback verification"),
        ..Default::default()
    }))
    .unwrap();
    let mut r = Renderer::new(device, queue, 28, 19, 42).unwrap();
    assert!(!r.hardware_ray_tracing_available());
    let s = Settings {
        spp: 3,
        ..Default::default()
    };
    r.accumulate(&s);
    let a = r.read_pixels().unwrap();
    r.reset();
    r.accumulate(&Settings {
        hardware_ray_tracing: false,
        ..s.clone()
    });
    assert_pixels(
        &a,
        &r.read_pixels().unwrap(),
        "Optional ray-query feature fallback",
    );
    for photon in [false, true] {
        let s = Settings {
            mode: Mode::WorldA,
            neural_cache: !photon,
            photon_nrc: photon,
            spp: 1,
            ..s.clone()
        };
        r.reset();
        r.accumulate(&s);
        assert!(
            r.read_linear_image(&s, false)
                .unwrap()
                .iter()
                .flatten()
                .all(|v| v.is_finite())
        );
    }
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn trace_workgroup_variants_preserve_partial_tiles_and_reconstruction_inputs() {
    let mut reference: Option<Vec<Vec<Pixel>>> = None;
    for group_y in [8, 4, 16, 32] {
        let (device, queue) = pollster::block_on(headless_device()).unwrap();
        let mut r = Renderer::new_with_workgroup(device, queue, 28, 19, 42, group_y).unwrap();
        let mut images = Vec::new();
        for mode in [Mode::WorldA, Mode::Blend, Mode::PerBounce] {
            for denoise in [false, true] {
                let s = Settings {
                    mode,
                    spp: 3,
                    denoise,
                    denoiser: crate::settings::DenoiserKind::Temporal,
                    ..Default::default()
                };
                r.reset();
                r.render(&s, true);
                images.push(r.read_pixels().unwrap());
            }
        }
        if let Some(reference) = &reference {
            for (a, b) in images.iter().zip(reference) {
                assert_pixels(a, b, "Trace workgroup and partial tile equivalence");
            }
        } else {
            reference = Some(images);
        }
    }
}

#[test]
#[ignore = "requires Metal/Vulkan/DX12 compute adapter"]
fn adaptive_large_image_preserves_budget_across_scan_chunks() {
    let (device, queue) = pollster::block_on(headless_device()).unwrap();
    let mut r = Renderer::new(device, queue, 512, 384, 42).unwrap();
    r.set_worlds(&[vec![], vec![]]).unwrap();
    let s = Settings {
        mode: Mode::WorldA,
        sampling: crate::settings::SamplingMode::Variance,
        spp: 1,
        ..Default::default()
    };
    for _ in 0..16 {
        r.accumulate(&s);
    }
    let mut pixels = r.read_pixels().unwrap();
    for (i, p) in pixels.iter_mut().enumerate() {
        p.moment_a[3] = if i % 512 < 256 { 1600. } else { 0. };
        p.stats_a[3] = 16.;
    }
    r.queue
        .write_buffer(&r.pixels, 0, bytemuck::cast_slice(&pixels));
    // At this real progressive index, jitter=.9977968. Adding it to the
    // 196608-ray endpoint rounds to 196609 in f32. Replay that phase without
    // tracing the intervening 145 rounds; physical path counts stay separate.
    r.samples = 161;
    for round in 0..24 {
        r.accumulate(&s);
        let pixels = r.read_pixels().unwrap();
        let schedules = pixels.iter().map(|p| u64::from(p.schedule[0])).sum::<u64>();
        let actual = pixels.iter().map(|p| u64::from(p.counts[0])).sum::<u64>();
        let expected = pixels.len() as u64 * (17 + round as u64);
        let guards = pixels.iter().map(|p| u64::from(p.counts[2])).sum::<u64>();
        assert_eq!(
            schedules,
            pixels.len() as u64,
            "round {round}: allocation sum, actual camera paths {actual}, expected {expected}, guards {guards}"
        );
        assert_eq!(
            actual, expected,
            "round {round}: compaction/tracing lost paths; guards {guards}"
        );
    }
}
