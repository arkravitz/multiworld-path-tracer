override DENOISE_INPUT: bool = false;
override ACTIVE_MODE: u32 = 4u;
override ADAPTIVE_TRACE:bool=false;
struct Params {
    size: vec4<u32>, // width, height, samples this dispatch, previous samples
    transport: vec4<u32>, // mode, max depth, seed, debug view
    controls: vec4<f32>, // p, persistence, exposure stops, query/display bitmask (1/2 BVH worlds, 4 fast linear, 8 fused display)
    flags: vec4<u32>, // spatial probability, primary A, world A count, world B count
    acceleration: vec4<u32>, // node counts A/B, unbounded plane counts A/B
    lights: vec4<u32>, // finite light counts A/B, MIS enabled, worlds with glass bitmask
    neural: vec4<u32>, // enabled, first cache bounce, training batch, self-training
    reconstruction:vec4<f32>, // OIDN strength, allocation policy, uniform warmup epochs, reserved
    noise: vec4<f32>, // threshold, frequency, softness, time offset
    eye: vec4<f32>, forward: vec4<f32>, right: vec4<f32>, up: vec4<f32>,
}
struct Object { center: vec4<f32>, extent: vec4<f32>, color: vec4<f32>, emission: vec4<f32> }
struct Node { low: vec3<f32>, escape: u32, high: vec3<f32>, object: u32 }
// Scan metadata has independent scalar memory locations. Different workgroups
// publish different members of the same record; vector components would alias
// under the shader memory model and permit lost updates on Metal.
struct Allocation {x:f32,y:f32,z:f32,w:f32}
struct Schedule {x:u32,y:u32,z:u32,w:u32}
struct Pixel { a: vec4<f32>, b: vec4<f32>, stats_a: vec4<f32>, stats_b: vec4<f32>, normals_a: vec4<f32>, normals_b: vec4<f32>, guide_normal_a:vec4<f32>, guide_albedo_a:vec4<f32>, guide_normal_b:vec4<f32>, guide_albedo_b:vec4<f32>, moment_a:vec4<f32>, moment_b:vec4<f32>, light_a:vec4<f32>, light_b:vec4<f32>, allocation:Allocation, work:vec4<u32>, schedule:Schedule, half_a:vec4<f32>, half_b:vec4<f32>, counts:vec4<u32> }
struct Frame { sample: vec4<f32>, position_a: vec4<f32>, normal_a: vec4<f32>, albedo_a: vec4<f32>, position_b: vec4<f32>, normal_b: vec4<f32>, albedo_b: vec4<f32> }
struct Hit { t: f32, normal: vec3<f32>, color: vec3<f32>, emission: vec3<f32>, metal: f32, light_pdf: f32, front: f32 }
struct Result { radiance: vec3<f32>, stats: vec3<f32>, normal: vec3<f32>, guide_normal:vec3<f32>, guide_albedo:vec3<f32> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> world_a: array<Object>;
@group(0) @binding(2) var<storage, read> world_b: array<Object>;
@group(0) @binding(3) var<storage, read_write> pixels: array<Pixel>;
@group(0) @binding(4) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var<storage, read> bvh_a: array<Node>;
@group(0) @binding(6) var<storage, read> bvh_b: array<Node>;
@group(0) @binding(7) var<storage, read_write> frame_store: FrameStore;

// One-dimensional randomized base-2 sequence for the initial world decision.
// Only that decision is stratified: independent RNGs still drive geometry and
// later Markov transitions. Absolute sample indices preserve dispatch invariance.
var<private> sequence_sample: u32;
var<private> sequence_scramble: u32;
var<private> sequence_active: bool;
fn initial_world_random(state: ptr<function,u32>) -> f32 {
    let fallback = random(state); // preserve transition stream dimensions
    if !sequence_active { return fallback; }
    return f32((reverseBits(sequence_sample)^sequence_scramble)>>8u)*(1.0/16777216.0);
}
fn hash(v: u32) -> u32 {
    var x = v; x = (x ^ (x >> 16u)) * 0x21f0aaadu;
    x = (x ^ (x >> 15u)) * 0x735a2d97u; return x ^ (x >> 15u);
}
fn random(state: ptr<function, u32>) -> f32 {
    *state = *state + 0x9e3779b9u;
    return f32(hash(*state) >> 8u) * (1.0 / 16777216.0);
}
fn lattice(p: vec3<i32>) -> f32 {
    return f32(hash(bitcast<u32>(p.x) ^ hash(bitcast<u32>(p.y)) ^ hash(bitcast<u32>(p.z) + 0x85ebca6bu)) >> 8u) / 16777216.0;
}
fn noise3(p: vec3<f32>) -> f32 {
    let cell = vec3<i32>(floor(p)); let t = fract(p); let u = t*t*(3.0-2.0*t);
    let a = mix(lattice(cell), lattice(cell+vec3<i32>(1,0,0)), u.x);
    let b = mix(lattice(cell+vec3<i32>(0,1,0)), lattice(cell+vec3<i32>(1,1,0)), u.x);
    let c = mix(lattice(cell+vec3<i32>(0,0,1)), lattice(cell+vec3<i32>(1,0,1)), u.x);
    let d = mix(lattice(cell+vec3<i32>(0,1,1)), lattice(cell+vec3<i32>(1,1,1)), u.x);
    return mix(mix(a,b,u.y), mix(c,d,u.y), u.z);
}
fn probability(origin: vec3<f32>) -> f32 {
    if params.flags.x != 0u {
        let n = noise3(origin*params.noise.y + vec3<f32>(params.noise.w, params.noise.w*0.37, -params.noise.w*0.21));
        return smoothstep(params.noise.x-params.noise.z, params.noise.x+params.noise.z, n);
    }
    return params.controls.x;
}
fn choose(origin: vec3<f32>, state: ptr<function,u32>) -> u32 {
    return select(1u,0u,random(state) < probability(origin));
}
// Ellipsoid intersections are sphere intersections in scaled local coordinates.
fn ellipsoid_distance(o: Object, ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    let oc=(ro-o.center.xyz)/o.extent.xyz; let direction=rd/o.extent.xyz;
    let a=dot(direction,direction); let b=dot(oc,direction);
    let disc=b*b-a*(dot(oc,oc)-1.0);
    if disc < 0.0 { return 1e30; }
    var t=(-b-sqrt(disc))/a;
    if t<=0.001 { t=(-b+sqrt(disc))/a; }
    return select(1e30,t,t>0.001);
}
// Finite cylinder plus hemispherical caps, including inside-origin exits.
fn capsule_distance(o: Object, ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    let start = o.center.xyz-o.extent.xyz;
    let axis = 2.0*o.extent.xyz;
    let length2 = dot(axis,axis);
    let oc = ro-start;
    let radius2 = o.center.w*o.center.w;
    var closest = 1e30;
    if length2 > 1e-12 {
        let along = dot(axis,rd); let projected = dot(axis,oc);
        let a = 1.0-along*along/length2;
        let b = dot(oc,rd)-projected*along/length2;
        let c = dot(oc,oc)-projected*projected/length2-radius2;
        let disc = b*b-a*c;
        if a > 1e-8 && disc >= 0.0 {
            let root = sqrt(disc);
            for (var side=0u;side<2u;side++) {
                let t = (-b+select(-root,root,side==1u))/a;
                let y = projected+t*along;
                if t > 0.001 && y >= 0.0 && y <= length2 { closest = min(closest,t); }
            }
        }
    }
    for (var end=0u;end<2u;end++) {
        let delta = ro-(start+f32(end)*axis);
        let b = dot(delta,rd); let disc = b*b-dot(delta,delta)+radius2;
        if disc >= 0.0 {
            let root = sqrt(disc);
            for (var side=0u;side<2u;side++) {
                let t = -b+select(-root,root,side==1u);
                let y = dot(ro+t*rd-start,axis);
                if t > 0.001 && (length2 <= 1e-12 || (end==0u && y<=0.0) || (end==1u && y>=length2)) { closest=min(closest,t); }
            }
        }
    }
    return closest;
}

// Two-sided Moller-Trumbore intersection. Tolerance closes tessellation seams;
// the geometric winding remains available for dielectric entry/exit decisions.
fn triangle_distance(o:Object,ro:vec3<f32>,rd:vec3<f32>)->f32 {
    let p=cross(rd,o.emission.xyz);let det=dot(o.extent.xyz,p);
    if abs(det)<1e-9 {return 1e30;}
    let v=ro-o.center.xyz;let u=dot(v,p)/det;let q=cross(v,o.extent.xyz);
    let w=dot(rd,q)/det;let t=dot(o.emission.xyz,q)/det;
    return select(1e30,t,u>=-1e-6 && w>=-1e-6 && u+w<=1.000001 && t>0.001);
}
fn primitive(o: Object, ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    var t = 1e30; var normal = vec3<f32>(0.0);
    if o.extent.w < 0.5 {
        let oc = ro-o.center.xyz; let b = dot(oc,rd);
        let disc = b*b-dot(oc,oc)+o.center.w*o.center.w;
        if disc >= 0.0 {
            var root = -b-sqrt(disc);
            if root <= 0.001 { root = -b+sqrt(disc); }
            if root > 0.001 { t = root; normal = (ro+rd*t-o.center.xyz)/o.center.w; }
        }
    } else if o.extent.w < 1.5 {
        let lo = o.center.xyz-o.extent.xyz; let hi = o.center.xyz+o.extent.xyz;
        var near = -1e30; var far = 1e30; var nn = vec3<f32>(0.0); var fnorm = nn;
        var valid = true;
        for (var axis = 0u; axis < 3u; axis++) {
            if abs(rd[axis]) < 1e-8 {
                if ro[axis] < lo[axis] || ro[axis] > hi[axis] { valid = false; }
            } else {
                var a = (lo[axis]-ro[axis])/rd[axis]; var b = (hi[axis]-ro[axis])/rd[axis];
                var n = vec3<f32>(0.0); n[axis] = -1.0;
                if a > b { let temp = a; a = b; b = temp; n = -n; }
                if a > near { near = a; nn = n; }
                if b < far { far = b; fnorm = -n; }
            }
        }
        if valid && near <= far && far > 0.001 {
            t = far; normal = fnorm;
            if near > 0.001 { t = near; normal = nn; }
        }
    } else if o.extent.w < 2.5 {
        if abs(rd.y) > 1e-8 {
            let root = (o.center.y-ro.y)/rd.y;
            if root > 0.001 { t = root; normal = vec3<f32>(0.0,1.0,0.0); }
        }
    } else if o.extent.w < 3.5 {
        t = capsule_distance(o,ro,rd);
        if t < 1e29 {
            let start = o.center.xyz-o.extent.xyz; let axis = 2.0*o.extent.xyz;
            let u = clamp(dot(ro+t*rd-start,axis)/max(1e-12,dot(axis,axis)),0.0,1.0);
            normal = normalize(ro+t*rd-start-u*axis);
        }
    }
    else if o.extent.w<4.5 {
        t=ellipsoid_distance(o,ro,rd);
        if t<1e29 {normal=normalize((ro+t*rd-o.center.xyz)/(o.extent.xyz*o.extent.xyz));}
    }
    else {
        t=triangle_distance(o,ro,rd);
        if t<1e29 {normal=normalize(cross(o.extent.xyz,o.emission.xyz));}
    }
    let front=select(0.0,1.0,dot(normal,rd)<=0.0);
    // Two-sided surfaces, including when switching places the origin inside an object.
    if dot(normal,rd) > 0.0 { normal = -normal; }
    var pdf = 0.0;
    if params.lights.z != 0u && t < 1e29 { pdf = emitter_pdf(o,ro,rd,t,normal); }
    return Hit(t,normal,o.color.xyz,select(o.emission.xyz,vec3<f32>(0.0),o.extent.w>4.5),o.color.w,pdf,front);
}
// Retained as an independently traversed correctness and performance reference.
fn intersect_brute(world: u32, ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    var closest = Hit(1e30,vec3<f32>(0.0),vec3<f32>(0.0),vec3<f32>(0.0),0.0,0.0,0.0);
    let count = select(params.flags.z,params.flags.w,world==1u);
    for (var i=0u; i<count; i++) {
        var o: Object;
        if world == 0u { o = world_a[i]; } else { o = world_b[i]; }
        let hit = primitive(o,ro,rd);
        if hit.t < closest.t { closest = hit; }
    }
    return closest;
}
// Candidate queries need only geometry and distance. Fetch normals/materials
// once for the winner, using the reference primitive to preserve edge conventions.
fn primitive_distance(o: Object, ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    if o.extent.w < 0.5 {
        let oc = ro-o.center.xyz; let b = dot(oc,rd);
        let disc = b*b-dot(oc,oc)+o.center.w*o.center.w;
        if disc < 0.0 { return 1e30; }
        let root_disc = sqrt(disc);
        var root = -b-root_disc;
        if root <= 0.001 { root = -b+root_disc; }
        return select(1e30,root,root>0.001);
    }
    if o.extent.w < 1.5 {
        let lo = o.center.xyz-o.extent.xyz; let hi = o.center.xyz+o.extent.xyz;
        var near = -1e30; var far = 1e30;
        for (var axis=0u; axis<3u; axis++) {
            if abs(rd[axis]) < 1e-8 {
                if ro[axis] < lo[axis] || ro[axis] > hi[axis] { return 1e30; }
            } else {
                let a = (lo[axis]-ro[axis])/rd[axis]; let b = (hi[axis]-ro[axis])/rd[axis];
                near = max(near,min(a,b)); far = min(far,max(a,b));
                if near > far { return 1e30; }
            }
        }
        if far <= 0.001 { return 1e30; }
        return select(far,near,near>0.001);
    }
    if o.extent.w > 4.5 {return triangle_distance(o,ro,rd);}
    if o.extent.w > 3.5 { return ellipsoid_distance(o,ro,rd); }
    if o.extent.w > 2.5 { return capsule_distance(o,ro,rd); }
    if abs(rd.y) > 1e-8 {
        let root = (o.center.y-ro.y)/rd.y;
        if root > 0.001 { return root; }
    }
    return 1e30;
}
fn object_at(world: u32, index: u32) -> Object {
    if world == 0u { return world_a[index]; }
    return world_b[index];
}
fn node_at(world: u32, index: u32) -> Node {
    if world == 0u { return bvh_a[index]; }
    return bvh_b[index];
}
fn bounds_hit(node: Node, ro: vec3<f32>, rd: vec3<f32>, inv: vec3<f32>, limit: f32) -> bool {
    var near = 0.0;
    // Conservative interval: tie hits must not be culled by rounding.
    var far = limit*1.000002 + 0.0001;
    for (var axis=0u; axis<3u; axis++) {
        // Avoid 0*infinity NaNs for exactly parallel and boundary-origin rays.
        if rd[axis] == 0.0 {
            if ro[axis] < node.low[axis] || ro[axis] > node.high[axis] { return false; }
        } else if abs(rd[axis]) >= 1e-8 || ro[axis] < node.low[axis] || ro[axis] > node.high[axis] {
            // The box primitive treats |direction|<1e-8 as parallel. If the
            // origin is inside this slab, leaving it unconstrained is conservative
            // for both that convention and exact sphere intersections.
            let a = (node.low[axis]-ro[axis])*inv[axis];
            let b = (node.high[axis]-ro[axis])*inv[axis];
            near = max(near,min(a,b)); far = min(far,max(a,b));
        }
    }
    return near <= far;
}
fn intersect_bvh(world: u32, ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    var closest = Hit(1e30,vec3<f32>(0.0),vec3<f32>(0.0),vec3<f32>(0.0),0.0,0.0,0.0);
    var nearest_id = 0xffffffffu;
    let plane_count = select(params.acceleration.z,params.acceleration.w,world==1u);
    let node_count = select(params.acceleration.x,params.acceleration.y,world==1u);
    // Infinite planes do not have finite AABBs. They also provide an early t cap.
    for (var i=0u; i<plane_count; i++) {
        let index = node_at(world,i).object;
        let t = primitive_distance(object_at(world,index),ro,rd);
        if t < closest.t || (t < 1e30 && t == closest.t && index < nearest_id) {
            closest.t = t; nearest_id = index;
        }
    }
    let inv = 1.0/select(rd,vec3<f32>(1.0),rd==vec3<f32>(0.0));
    // Fresh root and interval for EVERY segment, including cross-world inside exits.
    var cursor = plane_count;
    loop {
        if cursor >= node_count { break; }
        let node = node_at(world,cursor);
        if !bounds_hit(node,ro,rd,inv,closest.t) { cursor = node.escape; continue; }
        if node.object == 0xffffffffu { cursor++; continue; }
        let index = node.object;
        let t = primitive_distance(object_at(world,index),ro,rd);
        // Preserve the linear query's original-index ordering at equal distance.
        if t < closest.t || (t < 1e30 && t == closest.t && index < nearest_id) {
            closest.t = t; nearest_id = index;
        }
        cursor = node.escape;
    }
    if nearest_id != 0xffffffffu { return primitive(object_at(world,nearest_id),ro,rd); }
    return closest;
}
fn intersect_linear_fast(world: u32, ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    var nearest = 1e30; var nearest_id = 0xffffffffu;
    let count = select(params.flags.z,params.flags.w,world==1u);
    for (var i=0u; i<count; i++) {
        let t = primitive_distance(object_at(world,i),ro,rd);
        if t < nearest { nearest = t; nearest_id = i; }
    }
    if nearest_id != 0xffffffffu { return primitive(object_at(world,nearest_id),ro,rd); }
    return Hit(1e30,vec3<f32>(0.0),vec3<f32>(0.0),vec3<f32>(0.0),0.0,0.0,0.0);
}
fn intersect(world: u32, ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    if (u32(params.controls.w) & (1u << world)) != 0u { return intersect_bvh(world,ro,rd); }
    if (u32(params.controls.w) & 4u) != 0u { return intersect_linear_fast(world,ro,rd); }
    return intersect_brute(world,ro,rd);
}
// A glass-only query reconstructs the incident medium for camera-inside rays
// and after a world switch. Closed, non-overlapping solids have an exit as their
// nearest boundary when the ray origin is inside their volume.
fn glass_boundary(world:u32,ro:vec3<f32>,rd:vec3<f32>)->Hit {
    var nearest=1e30;var nearest_id=0xffffffffu;
    if (u32(params.controls.w)&(1u<<world))!=0u {
        let count=select(params.acceleration.x,params.acceleration.y,world==1u);
        var cursor=select(params.acceleration.z,params.acceleration.w,world==1u);
        let inv=1.0/select(rd,vec3<f32>(1.0),rd==vec3<f32>(0.0));
        loop {
            if cursor>=count {break;}
            let node=node_at(world,cursor);
            if !bounds_hit(node,ro,rd,inv,nearest) {cursor=node.escape;continue;}
            if node.object!=0xffffffffu {
                let o=object_at(world,node.object);
                if o.color.w>=2.0 {let t=primitive_distance(o,ro,rd);if t<nearest {nearest=t;nearest_id=node.object;}}
            }
            cursor++;
        }
    }else {
        let count=select(params.flags.z,params.flags.w,world==1u);
        for(var i=0u;i<count;i++) {
            let o=object_at(world,i);if o.color.w<2.0 {continue;}
            let t=primitive_distance(o,ro,rd);if t<nearest {nearest=t;nearest_id=i;}
        }
    }
    if nearest_id!=0xffffffffu {return primitive(object_at(world,nearest_id),ro,rd);}
    return Hit(1e30,vec3<f32>(0.0),vec3<f32>(0.0),vec3<f32>(0.0),0.0,0.0,0.0);
}
fn dielectric_fresnel(cosine:f32,eta_i:f32,eta_t:f32)->f32 {
    if abs(eta_i-eta_t)<1e-6 {return 0.0;}
    let sin2_t=(eta_i/eta_t)*(eta_i/eta_t)*max(0.0,1.0-cosine*cosine);
    if sin2_t>=1.0 {return 1.0;}
    let ct=sqrt(max(0.0,1.0-sin2_t));
    let parallel=(eta_t*cosine-eta_i*ct)/(eta_t*cosine+eta_i*ct);
    let perpendicular=(eta_i*cosine-eta_t*ct)/(eta_i*cosine+eta_t*ct);
    return 0.5*(parallel*parallel+perpendicular*perpendicular);
}
const PI: f32 = 3.14159265359;
fn power_weight(a: f32, b: f32) -> f32 {
    // Ratio form avoids squaring large solid-angle densities.
    if a <= 0.0 { return 0.0; }
    let r = b/a;
    return 1.0/(1.0+r*r);
}
fn sphere_cone(o: Object, ro: vec3<f32>) -> f32 {
    let ratio = min(1.0,o.center.w*o.center.w/dot(o.center.xyz-ro,o.center.xyz-ro));
    return ratio/(1.0+sqrt(1.0-ratio)); // stable 1-cos(theta_max)
}
fn box_face_areas(o: Object, ro: vec3<f32>) -> vec3<f32> {
    let d = abs(ro-o.center.xyz);
    let inside = all(d < o.extent.xyz);
    let areas = 4.0*o.extent.yzx*o.extent.zxy;
    return areas*select(select(vec3<f32>(0.0),vec3<f32>(1.0),d>=o.extent.xyz),vec3<f32>(2.0),inside);
}
fn emitter_pdf(o: Object, ro: vec3<f32>, rd: vec3<f32>, t: f32, normal: vec3<f32>) -> f32 {
    if o.emission.w <= 0.0 { return 0.0; }
    if o.extent.w < 0.5 {
        if distance(ro,o.center.xyz) <= o.center.w { return o.emission.w/(4.0*PI); }
        return o.emission.w/(2.0*PI*sphere_cone(o,ro));
    }
    let areas = box_face_areas(o,ro);
    let area = areas.x+areas.y+areas.z;
    return o.emission.w*t*t/max(1e-20,area*abs(dot(normal,rd)));
}
fn oriented_direction(n: vec3<f32>, cos_theta: f32, phi: f32) -> vec3<f32> {
    let helper = select(vec3<f32>(0.0,0.0,1.0),vec3<f32>(1.0,0.0,0.0),abs(n.z)>0.999);
    let tangent = normalize(cross(helper,n)); let bitangent = cross(n,tangent);
    let sin_theta = sqrt(max(0.0,1.0-cos_theta*cos_theta));
    return normalize(tangent*(sin_theta*cos(phi)) + bitangent*(sin_theta*sin(phi)) + n*cos_theta);
}
// Early-channel visibility, bounded at the sampled light. No shading for occluders.
fn shadow_occluded(world: u32, ro: vec3<f32>, rd: vec3<f32>, limit: f32) -> bool {
    if (u32(params.controls.w) & (1u << world)) == 0u {
        let count = select(params.flags.z,params.flags.w,world==1u);
        for (var i=0u; i<count; i++) {
            if primitive_distance(object_at(world,i),ro,rd) < limit { return true; }
        }
        return false;
    }
    let planes = select(params.acceleration.z,params.acceleration.w,world==1u);
    let count = select(params.acceleration.x,params.acceleration.y,world==1u);
    for (var i=0u; i<planes; i++) {
        if primitive_distance(object_at(world,node_at(world,i).object),ro,rd) < limit { return true; }
    }
    let inv = 1.0/select(rd,vec3<f32>(1.0),rd==vec3<f32>(0.0));
    var cursor = planes;
    loop {
        if cursor >= count { break; }
        let node = node_at(world,cursor);
        if !bounds_hit(node,ro,rd,inv,limit) { cursor = node.escape; continue; }
        if node.object != 0xffffffffu {
            if primitive_distance(object_at(world,node.object),ro,rd) < limit { return true; }
        }
        cursor++;
    }
    return false;
}
struct LightDirection { object: Object, direction: vec3<f32> }
fn sample_light_direction(world: u32, ro: vec3<f32>, state: ptr<function,u32>) -> LightDirection {
    let count = select(params.lights.x,params.lights.y,world==1u);
    let offset = select(params.acceleration.x,params.acceleration.y,world==1u);
    let u = random(state);
    var low = 0u; var high = count-1u;
    loop {
        if low >= high { break; }
        let mid = (low+high)/2u;
        if u < node_at(world,offset+mid).low.x { high = mid; } else { low = mid+1u; }
    }
    let o = object_at(world,node_at(world,offset+low).object);
    var direction: vec3<f32>;
    if o.extent.w < 0.5 {
        let delta = o.center.xyz-ro;
        if dot(delta,delta) <= o.center.w*o.center.w {
            direction = oriented_direction(vec3<f32>(0.0,1.0,0.0),1.0-2.0*random(state),2.0*PI*random(state));
        } else {
            direction = oriented_direction(normalize(delta),1.0-random(state)*sphere_cone(o,ro),2.0*PI*random(state));
        }
    } else {
        let areas = box_face_areas(o,ro);
        let total = areas.x+areas.y+areas.z;
        var face = random(state)*total;
        var axis = 0u;
        if face >= areas.x { face -= areas.x; axis = 1u;
            if face >= areas.y { face -= areas.y; axis = 2u; }
        }
        let inside = all(abs(ro-o.center.xyz) < o.extent.xyz);
        var side = select(-1.0,1.0,ro[axis]>=o.center[axis]);
        if inside { side = select(-1.0,1.0,face>=areas[axis]*0.5); }
        var point = o.center.xyz;
        point[axis] += side*o.extent[axis];
        let a = (axis+1u)%3u; let b = (axis+2u)%3u;
        point[a] += (2.0*random(state)-1.0)*o.extent[a];
        point[b] += (2.0*random(state)-1.0)*o.extent[b];
        direction = normalize(point-ro);
    }
    return LightDirection(o,direction);
}
fn light_direction_pdf(world: u32, ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    let count = select(params.lights.x,params.lights.y,world==1u);
    let offset = select(params.acceleration.x,params.acceleration.y,world==1u);
    var pdf = 0.0;
    // Sum all proposal densities, including lights behind occluders or other
    // emitters. Visibility belongs to transport, never to this directional PDF.
    for (var i=0u; i<count; i++) {
        let o = object_at(world,node_at(world,offset+i).object);
        let h = primitive(o,ro,rd);
        if h.t < 1e29 { pdf += h.light_pdf; }
    }
    return pdf;
}
fn direct_light(world: u32, ro: vec3<f32>, normal: vec3<f32>, state: ptr<function,u32>) -> vec3<f32> {
    let count = select(params.lights.x,params.lights.y,world==1u);
    if count == 0u { return vec3<f32>(0.0); }
    let sample = sample_light_direction(world,ro,state);
    let o = sample.object; let direction = sample.direction;
    let cosine = max(0.0,dot(normal,direction));
    if cosine <= 0.0 { return vec3<f32>(0.0); }
    let hit = primitive(o,ro,direction);
    if hit.t > 1e29 || hit.light_pdf <= 0.0 { return vec3<f32>(0.0); }
    if shadow_occluded(world,ro,direction,hit.t-max(0.0001,hit.t*0.00001)) { return vec3<f32>(0.0); }
    let bsdf_pdf = cosine/PI;
    return o.emission.xyz*(bsdf_pdf/hit.light_pdf)*power_weight(hit.light_pdf,bsdf_pdf);
}

fn cosine_direction(n: vec3<f32>, state: ptr<function,u32>) -> vec3<f32> {
    let u = random(state); let phi = 6.28318530718*random(state);
    let helper = select(vec3<f32>(0.0,0.0,1.0),vec3<f32>(1.0,0.0,0.0),abs(n.z)>0.999);
    let tangent = normalize(cross(helper,n)); let bitangent = cross(n,tangent);
    return normalize(tangent*(sqrt(u)*cos(phi)) + bitangent*(sqrt(u)*sin(phi)) + n*sqrt(1.0-u));
}
fn trace(ro_in: vec3<f32>, rd_in: vec3<f32>, seed: u32, mode: u32) -> Result {
    var training_prefix:array<vec3<f32>,4>;var training_scale:array<vec3<f32>,4>;var training_count=0u;
    var training_suffix=false;var training_bootstrapped=false;var training_terminal_scale=vec3<f32>(0.0);
    var light_rng = hash(seed ^ 0x7f4a7c15u);
    var guide_normal=vec3<f32>(0.0);var guide_albedo=vec3<f32>(1.0);var guide_weight=vec3<f32>(1.0);var guide_found=false;var guide_only=false;
    var previous_pdf = 0.0;
    var caustic_receiver=false;var caustic_chain=false;
    let switching = mode >= 4u && (params.flags.x != 0u || (params.controls.x > 0.0 && params.controls.x < 1.0)) && (mode != 5u || params.controls.y < 1.0);
    let directional_mis = params.lights.z != 0u && switching;
    var rng = seed; var selection_rng = hash(seed ^ 0xa511e9b3u);
    var ro = ro_in; var rd = rd_in;
    var throughput = vec3<f32>(1.0); var radiance = vec3<f32>(0.0);
    var medium_world=2u;var medium_ior=1.0;var medium_active=false;var medium_color=vec3<f32>(1.0);var eta_scale=1.0;
    var camera_footprint=1.0;var path_spread=0.0;var incoming_pdf=0.0;
    var world = 0u;
    var primary_a: Hit; var primary_b: Hit; var cached_primary = false;
    if mode == 1u { world = 1u; }
    if mode >= 3u {
        if params.flags.x != 0u && params.flags.y == 0u {
            // The camera origin is shared by every primary ray. Sampling noise
            // there made the ENTIRE visible world flip when the camera orbited.
            // Use the first surface of the shared union as a world-space anchor.
            primary_a = intersect(0u,ro,rd); primary_b = intersect(1u,ro,rd);
            cached_primary = true;
            let distance = min(primary_a.t,primary_b.t);
            var p = params.controls.x; // Empty space has no surface anchor.
            if distance < 1e29 { p = probability(ro+distance*rd); }
            world = select(1u,0u,initial_world_random(&selection_rng)<p);
        } else { world = select(1u,0u,initial_world_random(&selection_rng)<probability(ro)); }
    }
    if mode >= 3u && params.flags.y != 0u { world = 0u; }
    let primary = world;
    var pending_switch = false;
    var switches = 0u; var bounces = 0u; var primary_normal = vec3<f32>(0.0);
    for (var bounce=0u; bounce<params.transport.y; bounce++) {
        if pending_switch && !guide_only { switches++; }
        pending_switch = false;
        if world!=medium_world {
            medium_world=world;medium_ior=1.0;medium_active=false;medium_color=vec3<f32>(1.0);
            if (params.lights.w&(1u<<world))!=0u {
                let boundary=glass_boundary(world,ro,rd);
                if boundary.t<1e29 && boundary.front<0.5 {medium_ior=boundary.metal-2.0;medium_active=true;medium_color=boundary.color;}
            }
        }
        var hit: Hit;
        if bounce == 0u && cached_primary {
            if world == 0u { hit = primary_a; } else { hit = primary_b; }
        } else { hit = intersect(world,ro,rd); }
        if hit.t > 1e29 {
            radiance += throughput*vec3<f32>(0.045,0.055,0.075); break;
        }
        if hit.metal>=2.0 && hit.front<0.5 {medium_ior=hit.metal-2.0;medium_active=true;medium_color=hit.color;}
        if medium_active {
            let absorption=pow(clamp(medium_color,vec3<f32>(0.0),vec3<f32>(1.0)),vec3<f32>(hit.t));
            throughput*=absorption;if !guide_found {guide_weight*=absorption;}
        }
        if !guide_found && hit.metal<=0.5 {
            guide_normal=hit.normal;guide_albedo=clamp(guide_weight*hit.color,vec3<f32>(0.0),vec3<f32>(1.0));guide_found=true;
        }
        if !guide_only {bounces++;}
        if NRC_ENABLED {
            let cosine=max(1e-5,abs(dot(hit.normal,rd)));
            if bounce==0u {camera_footprint=hit.t*hit.t/(4.0*PI*cosine);}
            else if incoming_pdf>1e-8 {path_spread+=hit.t/sqrt(incoming_pdf*cosine);}
        }
        if guide_only && guide_found {break;}
        if bounce == 0u { primary_normal = hit.normal*0.5+0.5; }
        var emission_weight = 1.0;
        if previous_pdf > 0.0 && hit.light_pdf > 0.0 { emission_weight = power_weight(previous_pdf,hit.light_pdf); }
        // Light tracing owns D S+ L paths; the camera still owns the sky,
        // direct lights, all-delta camera paths, and every diffuse continuation.
        if !(CAUSTICS_ENABLED && caustic_receiver && caustic_chain && hit.light_pdf>0.0) {
            radiance += throughput*hit.emission*emission_weight;
        }
        if hit.metal<2.0 && max(hit.color.x,max(hit.color.y,hit.color.z)) == 0.0 { break; }
        let position = ro+hit.t*rd;
        if CAUSTICS_ENABLED && hit.metal<=0.5 && bounce+1u<params.transport.y {
            radiance+=throughput*caustic_estimate(world,position,hit.normal,hit.color,bounce,rd_in);
            caustic_receiver=true;caustic_chain=false;
        } else if CAUSTICS_ENABLED && hit.metal>0.5 && caustic_receiver {caustic_chain=true;}
        if NRC_ENABLED && hit.metal<=0.5 && bounce+1u<params.transport.y {
            let sample=NeuralSample(vec4<f32>(position+hit.normal*0.002,f32(world)),vec4<f32>(hit.normal,f32(params.transport.y-bounce-1u)),vec4<f32>(hit.color,0.0),vec4<f32>(0.0));
            let eligible=select(bounce>=params.neural.y,bounce>0u && path_spread*path_spread>0.01*camera_footprint,params.neural.y==0u);
            if NRC_TEACHER && neural_teacher {
                if NRC_BOOTSTRAP && params.neural.w!=0u && !neural_uncached_anchor && neural_ready() && eligible {
                    // A second spread event closes the training suffix. The
                    // frozen training model predicts only scattering at this vertex;
                    // its local emission was already added above, exactly once.
                    if training_suffix {
                        if NRC_TEACHER_QUEUED {
                            var query=sample;query.label.w=1.0;
                            frame_store.neural.teacher_queries[neural_record_base/4u]=query;
                            training_terminal_scale=throughput*hit.color;
                        }else {radiance+=throughput*hit.color*neural_predict(sample);}
                        training_bootstrapped=true;break;
                    }
                    training_suffix=true;path_spread=0.0;
                }
                // The terminal query is deliberately excluded from supervision:
                // only preceding vertices learn from its transported prediction.
                if training_count<4u {
                    frame_store.neural.samples[neural_record_base+training_count]=sample;
                    training_prefix[training_count]=radiance;training_scale[training_count]=throughput*hit.color;training_count++;
                }
            }
            if !NRC_TEACHER && params.neural.x!=0u && eligible && neural_ready() {
                if NRC_COOPERATIVE || NRC_QUEUED {neural_pending=true;neural_pending_sample=sample;neural_pending_scale=throughput*hit.color;}
                else {radiance+=throughput*hit.color*neural_predict(sample);}
                break;
            }
        }

        previous_pdf = 0.0;
        var transmitted=false;
        if hit.metal>=2.0 {
            let eta_i=select(hit.metal-2.0,medium_ior,hit.front>0.5);
            let eta_t=select(1.0,hit.metal-2.0,hit.front>0.5);
            let cosine=clamp(-dot(rd,hit.normal),0.0,1.0);
            let fresnel=dielectric_fresnel(cosine,eta_i,eta_t);
            if random(&rng)<fresnel {rd=reflect(rd,hit.normal);}
            else {
                let eta=eta_i/eta_t;
                rd=normalize(eta*rd+(eta*cosine-sqrt(max(0.0,1.0-eta*eta*(1.0-cosine*cosine))))*hit.normal);
                throughput*=eta*eta;eta_scale=min(1e20,eta_scale/(eta*eta));
                medium_ior=eta_t;medium_active=hit.front>0.5;medium_color=select(vec3<f32>(1.0),hit.color,hit.front>0.5);
                transmitted=true;
            }
        }else {
            if hit.metal>0.5 {rd=reflect(rd,hit.normal);}else {rd=cosine_direction(hit.normal,&rng);}
            throughput*=hit.color;
            if !guide_found {guide_weight*=hit.color;}
        }
        incoming_pdf=select(max(0.0,dot(hit.normal,rd))/PI,0.0,hit.metal>0.5);
        ro=position+select(0.002,-0.002,transmitted)*hit.normal;
        if bounce >= 3u && !guide_only {
            let survival = clamp(max(throughput.x,max(throughput.y,throughput.z))*eta_scale,0.05,0.95);
            if random(&rng) >= survival {
                if guide_found {break;}
                // Preserve the terminated radiance and statistics, but finish
                // the delta guide path without roulette. This prevents dark
                // guide holes in multi-interface glass at low sample counts.
                guide_only=true;throughput=vec3<f32>(0.0);
            }else {throughput /= survival;}
        }
        // NEE and continuation condition on the SAME next segment world. The
        // transition probability is part of the transport process, not a PDF
        // to divide channel. Selection happens after roulette, as in the reference.
        if bounce+1u >= params.transport.y { break; }
        if mode >= 4u {
            let previous = world;
            var resample = mode == 4u || params.controls.y <= 0.0;
            if mode == 5u && params.controls.y > 0.0 && params.controls.y < 1.0 {
                resample = random(&selection_rng) >= params.controls.y;
            }
            if resample { world = choose(ro,&selection_rng); }
            pending_switch = world != previous;
        }
        if params.lights.z != 0u && hit.metal <= 0.5 {
            if directional_mis {
                let count = select(params.lights.x,params.lights.y,world==1u);
                if count > 0u {
                    // One-sample balance MIS: use one continuing ray instead of
                    // a separate shadow ray in the expensive stochastic modes.
                    if random(&light_rng) < 0.5 { rd = sample_light_direction(world,ro,&light_rng).direction; }
                    let cosine_pdf = max(0.0,dot(hit.normal,rd))/PI;
                    if cosine_pdf <= 0.0 { break; }
                    let mixture_pdf = 0.5*(cosine_pdf+light_direction_pdf(world,ro,rd));
                    incoming_pdf=mixture_pdf;
                    throughput *= cosine_pdf/mixture_pdf;
                }
            } else {
                previous_pdf = max(0.0,dot(hit.normal,rd))/PI;
                radiance += throughput*direct_light(world,ro,hit.normal,&light_rng);
            }
        }
    }
    if NRC_ENABLED && NRC_TEACHER && neural_teacher {
        for (var i=0u;i<training_count;i++) {
            let scale=training_scale[i];
            let label=max(vec3<f32>(0.0),(radiance-training_prefix[i])/max(scale,vec3<f32>(1e-6)));
            let valid=select(0.0,1.0,all(scale>vec3<f32>(1e-6)) && all(label<vec3<f32>(1e8)));
            frame_store.neural.samples[neural_record_base+i].label=vec4<f32>(label,valid);
            frame_store.neural.samples[neural_record_base+i].albedo.w=select(select(0.0,1.0,training_bootstrapped),2.0,neural_uncached_anchor);
            if NRC_TEACHER_QUEUED && training_bootstrapped {
                frame_store.neural.bootstrap_scales[neural_record_base+i]=vec4<f32>(training_terminal_scale/max(scale,vec3<f32>(1e-6)),0.0);
            }
        }
    }
    if !guide_found {guide_albedo=clamp(guide_weight,vec3<f32>(0.0),vec3<f32>(1.0));}
    return Result(radiance,vec3<f32>(select(0.0,1.0,primary==0u),f32(switches),f32(bounces)),primary_normal,guide_normal,guide_albedo);
}
fn linear_pixel(pixel: Pixel, count: f32) -> vec3<f32> {
    if params.transport.x == 2u { return mix(pixel.b.xyz,pixel.a.xyz,params.controls.x)/count; }
    return pixel.a.xyz/count;
}
fn guide_frame(id: vec3<u32>, color: vec3<f32>, count: f32) -> Frame {
    let uv=(vec2<f32>(id.xy)+0.5)/vec2<f32>(params.size.xy);
    let direction=normalize(params.forward.xyz+(2.0*uv.x-1.0)*params.right.xyz+(1.0-2.0*uv.y)*params.up.xyz);
    var frame: Frame; frame.sample=vec4<f32>(color,count);
    if params.transport.x != 1u {
        let a=intersect(0u,params.eye.xyz,direction);
        if a.t<1e29 {frame.position_a=vec4<f32>(params.eye.xyz+a.t*direction,1.0);frame.normal_a=vec4<f32>(a.normal,a.metal);frame.albedo_a=vec4<f32>(a.color,dot(a.emission,vec3<f32>(0.3,0.5,0.2)));}
    }
    if params.transport.x != 0u {
        let b=intersect(1u,params.eye.xyz,direction);
        if b.t<1e29 {frame.position_b=vec4<f32>(params.eye.xyz+b.t*direction,1.0);frame.normal_b=vec4<f32>(b.normal,b.metal);frame.albedo_b=vec4<f32>(b.color,dot(b.emission,vec3<f32>(0.3,0.5,0.2)));}
    }
    return frame;
}
fn accumulate_pixel(id:vec3<u32>,lane:u32) {
    let live=id.x<params.size.x && id.y<params.size.y;
    if !live && !NRC_COOPERATIVE {return;}
    let index=id.y*params.size.x+id.x;
    // Keep statistics in storage, not live in registers across ray traversal.
    // A full Pixel contains hundreds of bytes and would spill private memory.
    if live && params.size.w==0u && !ADAPTIVE_TRACE {pixels[index]=Pixel();}
    var sum_a=vec3<f32>(0.0);var sum_b=sum_a;var stats_a=sum_a;var stats_b=sum_a;var normals_a=sum_a;var normals_b=sum_a;
    if live && (params.size.w>0u || ADAPTIVE_TRACE) {
        sum_a=pixels[index].a.xyz;stats_a=pixels[index].stats_a.xyz;normals_a=pixels[index].normals_a.xyz;
        if ACTIVE_MODE==2u {sum_b=pixels[index].b.xyz;stats_b=pixels[index].stats_b.xyz;normals_b=pixels[index].normals_b.xyz;}
    }
    if NRC_QUEUED && live {neural_clear_queries(index);}
    var fresh_a=vec3<f32>(0.0);var fresh_b=vec3<f32>(0.0);
    let control=ADAPTIVE_TRACE && params.size.w>=u32(params.reconstruction.z);
    let predictor_a=select(vec3<f32>(0.0),pixels[index].moment_a.xyz,control);
    let predictor_b=select(vec3<f32>(0.0),pixels[index].moment_b.xyz,control);
    let old_stats_a=select(vec3<f32>(0.0),stats_a/f32(params.size.w+1u),control);
    let old_stats_b=select(vec3<f32>(0.0),stats_b/f32(params.size.w+1u),control);
    let old_normal_a=select(vec3<f32>(0.0),normals_a/f32(params.size.w+1u),control);
    let old_normal_b=select(vec3<f32>(0.0),normals_b/f32(params.size.w+1u),control);
    var samples=params.size.z;var scale=1.0;
    if ADAPTIVE_TRACE {
        // Bounded priorities request at most six rays. Guard corrupt metadata
        // separately from the allocation law so it cannot hang a GPU kernel.
        samples=min(pixels[index].schedule.x,8u);if pixels[index].schedule.x>8u {pixels[index].counts.z++;}
        scale=1.0/bitcast<f32>(pixels[index].schedule.y);
    }
    let first_sample=select(params.size.w,pixels[index].counts.x,ADAPTIVE_TRACE);
    let pixel_seed=index^hash(params.transport.z);
    for(var sample=0u;sample<samples;sample++) {
        sequence_sample=first_sample+sample;sequence_scramble=hash(pixel_seed^0xe7037ed1u);
        // An adaptive queue needs fresh world choices independent of its past
        // observations; uniform paths retain their stratified world sequence.
        sequence_active=params.lights.z!=0u && !ADAPTIVE_TRACE;
        let seed=hash(pixel_seed^hash(first_sample+sample));
        var camera_rng=hash(seed^0x63d83595u);
        let uv=(vec2<f32>(id.xy)+vec2<f32>(random(&camera_rng),random(&camera_rng)))/vec2<f32>(params.size.xy);
        let direction=normalize(params.forward.xyz+(2.0*uv.x-1.0)*params.right.xyz+(1.0-2.0*uv.y)*params.up.xyz);
        neural_pending=false;var a:Result;
        if live {a=trace(params.eye.xyz,direction,seed,select(ACTIVE_MODE,0u,ACTIVE_MODE==2u));}
        if NRC_QUEUED && neural_pending {
            neural_pending_sample.albedo.w=adaptive_luma(a.radiance);
            neural_pending_sample.label=vec4<f32>(neural_pending_scale,1.0);
            neural_enqueue(index*select(1u,2u,ACTIVE_MODE==2u),neural_pending_sample);
        }
        if NRC_COOPERATIVE {let prediction=neural_resolve_matrix(lane);if neural_pending {a.radiance+=neural_pending_scale*prediction;}}
        if live {observe_camera(index,a,0u);}
        if !ADAPTIVE_TRACE && live {
            let visible=caustic_visible(index,select(0u,1u,ACTIVE_MODE==1u));
            pixels[index].light_a+=vec4<f32>(visible,1.0);a.radiance+=visible;
        }
        if DENOISE_INPUT {fresh_a+=a.radiance;}
        sum_a+=(a.radiance-predictor_a)*scale;
        stats_a+=(a.stats-old_stats_a)*scale;
        normals_a+=(a.normal-old_normal_a)*scale;
        if ACTIVE_MODE==2u {
            neural_pending=false;var b:Result;if live {b=trace(params.eye.xyz,direction,seed,1u);}
            if NRC_QUEUED && neural_pending {
                neural_pending_sample.albedo.w=adaptive_luma(b.radiance);
                neural_pending_sample.label=vec4<f32>(neural_pending_scale,1.0);neural_enqueue(index*2u+1u,neural_pending_sample);
            }
            if NRC_COOPERATIVE {let prediction=neural_resolve_matrix(lane);if neural_pending {b.radiance+=neural_pending_scale*prediction;}}
            if live {observe_camera(index,b,1u);}
            if !ADAPTIVE_TRACE && live {let visible=caustic_visible(index,1u);pixels[index].light_b+=vec4<f32>(visible,1.0);b.radiance+=visible;}
            if DENOISE_INPUT {fresh_b+=b.radiance;}
            sum_b+=(b.radiance-predictor_b)*scale;
            stats_b+=(b.stats-old_stats_b)*scale;
            normals_b+=(b.normal-old_normal_b)*scale;
        }
    }
    if !live {return;}
    pixels[index].a=vec4<f32>(sum_a,pixels[index].a.w);pixels[index].stats_a=vec4<f32>(stats_a,pixels[index].stats_a.w);pixels[index].normals_a=vec4<f32>(normals_a,pixels[index].normals_a.w);
    if ACTIVE_MODE==2u {pixels[index].b=vec4<f32>(sum_b,pixels[index].b.w);pixels[index].stats_b=vec4<f32>(stats_b,pixels[index].stats_b.w);pixels[index].normals_b=vec4<f32>(normals_b,pixels[index].normals_b.w);}
    if DENOISE_INPUT {
        var fresh=select(fresh_a,mix(fresh_b,fresh_a,params.controls.x),ACTIVE_MODE==2u);
        let count=f32(params.size.z);fresh/=max(count,1.0);frame_store.frames[index]=guide_frame(id,fresh,count);
    }
    if (u32(params.controls.w)&8u)!=0u {display_pixel(id,pixels[index]);}
}
@compute @workgroup_size(8,TRACE_GROUP_Y)
fn accumulate(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32) {accumulate_pixel(id,lane);}
@compute @workgroup_size(64)
fn accumulate_adaptive(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    if id.x>=u32(pixels[0].allocation.x) {return;}
    let index=pixels[id.x].work.x;
    accumulate_pixel(vec3<u32>(index%params.size.x,index/params.size.x,0u),lane);
}
fn srgb(linear: vec3<f32>) -> vec3<f32> {
    return select(1.055*pow(max(linear,vec3<f32>(0.0)),vec3<f32>(1.0/2.4))-0.055,12.92*linear,linear<=vec3<f32>(0.0031308));
}
fn display_estimate(id: vec3<u32>, pixel: Pixel, estimate: vec3<f32>) {
    let count = f32(max(1u,params.size.w+params.size.z));
    var color = pixel.a.xyz/count; var stats = pixel.stats_a.xyz/count; var normals = pixel.normals_a.xyz/count;
    if params.transport.x == 2u {
        color = mix(pixel.b.xyz,pixel.a.xyz,params.controls.x)/count;
        stats = mix(pixel.stats_b.xyz,pixel.stats_a.xyz,params.controls.x)/count;
        normals = mix(pixel.normals_b.xyz,pixel.normals_a.xyz,params.controls.x)/count;
    }
    if params.transport.w == 0u {
        color = max(vec3<f32>(0.0),estimate);
        color *= exp2(params.controls.z);
        color = srgb(color/(vec3<f32>(1.0)+color)); // Reinhard, then exact sRGB encoding.
    } else if params.transport.w == 1u { color = mix(vec3<f32>(0.1,0.6,1.0),vec3<f32>(1.0,0.25,0.08),stats.x); }
    else if params.transport.w == 2u { color = vec3<f32>(stats.y/f32(max(1u,params.transport.y-1u)),0.1,1.0-stats.y/f32(max(1u,params.transport.y-1u))); }
    else if params.transport.w == 3u { color = vec3<f32>(stats.z/f32(params.transport.y)); }
    else if params.transport.w==5u {
        let ratio=f32(pixel.counts.x)/count;
        color=mix(vec3<f32>(0.04,0.2,0.7),vec3<f32>(1.0,0.25,0.03),clamp(ratio/2.0,0.0,1.0));
    }else { color = normals; }
    textureStore(output,vec2<i32>(id.xy),vec4<f32>(color,1.0));
}

fn display_pixel(id: vec3<u32>, pixel: Pixel) {
    display_estimate(id,pixel,linear_pixel(pixel,f32(max(1u,params.size.w+params.size.z))));
}
@compute @workgroup_size(8,8)
fn present(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.size.x || id.y >= params.size.y { return; }
    display_pixel(id,pixels[id.y*params.size.x+id.x]);
}
