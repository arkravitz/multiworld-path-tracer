override DENOISE_INPUT: bool = false;
override ACTIVE_MODE: u32 = 4u;
struct Params {
    size: vec4<u32>, // width, height, samples this dispatch, previous samples
    transport: vec4<u32>, // mode, max depth, seed, debug view
    controls: vec4<f32>, // p, persistence, exposure stops, query/display bitmask (1/2 BVH worlds, 4 fast linear, 8 fused display)
    flags: vec4<u32>, // spatial probability, primary A, world A count, world B count
    acceleration: vec4<u32>, // node counts A/B, unbounded plane counts A/B
    lights: vec4<u32>, // finite light counts A/B, MIS enabled, reserved
    noise: vec4<f32>, // threshold, frequency, softness, time offset
    eye: vec4<f32>, forward: vec4<f32>, right: vec4<f32>, up: vec4<f32>,
}
struct Object { center: vec4<f32>, extent: vec4<f32>, color: vec4<f32>, emission: vec4<f32> }
struct Node { low: vec3<f32>, escape: u32, high: vec3<f32>, object: u32 }
struct Pixel { a: vec4<f32>, b: vec4<f32>, stats_a: vec4<f32>, stats_b: vec4<f32>, normals_a: vec4<f32>, normals_b: vec4<f32> }
struct Frame { sample: vec4<f32>, position_a: vec4<f32>, normal_a: vec4<f32>, albedo_a: vec4<f32>, position_b: vec4<f32>, normal_b: vec4<f32>, albedo_b: vec4<f32> }
struct Hit { t: f32, normal: vec3<f32>, color: vec3<f32>, emission: vec3<f32>, metal: f32, light_pdf: f32 }
struct Result { radiance: vec3<f32>, stats: vec3<f32>, normal: vec3<f32> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> world_a: array<Object>;
@group(0) @binding(2) var<storage, read> world_b: array<Object>;
@group(0) @binding(3) var<storage, read_write> pixels: array<Pixel>;
@group(0) @binding(4) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var<storage, read> bvh_a: array<Node>;
@group(0) @binding(6) var<storage, read> bvh_b: array<Node>;
@group(0) @binding(7) var<storage, read_write> frames: array<Frame>;

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
    else {
        t=ellipsoid_distance(o,ro,rd);
        if t<1e29 {normal=normalize((ro+t*rd-o.center.xyz)/(o.extent.xyz*o.extent.xyz));}
    }
    // Two-sided surfaces, including when switching places the origin inside an object.
    if dot(normal,rd) > 0.0 { normal = -normal; }
    var pdf = 0.0;
    if params.lights.z != 0u && t < 1e29 { pdf = emitter_pdf(o,ro,rd,t,normal); }
    return Hit(t,normal,o.color.xyz,o.emission.xyz,o.color.w,pdf);
}
// Retained as an independently traversed correctness and performance reference.
fn intersect_brute(world: u32, ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    var closest = Hit(1e30,vec3<f32>(0.0),vec3<f32>(0.0),vec3<f32>(0.0),0.0,0.0);
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
    var closest = Hit(1e30,vec3<f32>(0.0),vec3<f32>(0.0),vec3<f32>(0.0),0.0,0.0);
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
    return Hit(1e30,vec3<f32>(0.0),vec3<f32>(0.0),vec3<f32>(0.0),0.0,0.0);
}
fn intersect(world: u32, ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    if (u32(params.controls.w) & (1u << world)) != 0u { return intersect_bvh(world,ro,rd); }
    if (u32(params.controls.w) & 4u) != 0u { return intersect_linear_fast(world,ro,rd); }
    return intersect_brute(world,ro,rd);
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
// Early-out visibility, bounded at the sampled light. No shading for occluders.
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
    var light_rng = hash(seed ^ 0x7f4a7c15u);
    var previous_pdf = 0.0;
    let switching = mode >= 4u && (params.flags.x != 0u || (params.controls.x > 0.0 && params.controls.x < 1.0)) && (mode != 5u || params.controls.y < 1.0);
    let directional_mis = params.lights.z != 0u && switching;
    var rng = seed; var selection_rng = hash(seed ^ 0xa511e9b3u);
    var ro = ro_in; var rd = rd_in;
    var throughput = vec3<f32>(1.0); var radiance = vec3<f32>(0.0);
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
        if pending_switch { switches++; }
        pending_switch = false;
        var hit: Hit;
        if bounce == 0u && cached_primary {
            if world == 0u { hit = primary_a; } else { hit = primary_b; }
        } else { hit = intersect(world,ro,rd); }
        if hit.t > 1e29 {
            radiance += throughput*vec3<f32>(0.045,0.055,0.075); break;
        }
        bounces++;
        if bounce == 0u { primary_normal = hit.normal*0.5+0.5; }
        var emission_weight = 1.0;
        if previous_pdf > 0.0 && hit.light_pdf > 0.0 { emission_weight = power_weight(previous_pdf,hit.light_pdf); }
        radiance += throughput*hit.emission*emission_weight;
        if max(hit.color.x,max(hit.color.y,hit.color.z)) == 0.0 { break; }
        let position = ro+hit.t*rd;
        previous_pdf = 0.0;
        if hit.metal > 0.5 { rd = reflect(rd,hit.normal); }
        else { rd = cosine_direction(hit.normal,&rng); }
        // Lambertian: f*cos/pdf = (albedo/pi)*cos/(cos/pi) = albedo.
        throughput *= hit.color;
        ro = position+hit.normal*0.002;
        if bounce >= 3u {
            let survival = clamp(max(throughput.x,max(throughput.y,throughput.z)),0.05,0.95);
            if random(&rng) >= survival { break; }
            throughput /= survival;
        }
        // NEE and continuation condition on the SAME next segment world. The
        // transition probability is part of the transport process, not a PDF
        // to divide out. Selection happens after roulette, as in the reference.
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
                    throughput *= cosine_pdf/mixture_pdf;
                }
            } else {
                previous_pdf = max(0.0,dot(hit.normal,rd))/PI;
                radiance += throughput*direct_light(world,ro,hit.normal,&light_rng);
            }
        }
    }
    return Result(radiance,vec3<f32>(select(0.0,1.0,primary==0u),f32(switches),f32(bounces)),primary_normal);
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
@compute @workgroup_size(8,8)
fn accumulate(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.size.x || id.y >= params.size.y { return; }
    let index = id.y*params.size.x+id.x;
    var pixel: Pixel;
    if params.size.w > 0u { pixel = pixels[index]; }
    var fresh_a=vec3<f32>(0.0); var fresh_b=vec3<f32>(0.0);
    let pixel_seed = index ^ hash(params.transport.z);
    for (var sample=0u; sample<params.size.z; sample++) {
        sequence_sample = params.size.w+sample;
        sequence_scramble = hash(pixel_seed ^ 0xe7037ed1u);
        sequence_active = params.lights.z != 0u;
        let seed = hash(pixel_seed ^ hash(params.size.w+sample));
        var camera_rng = hash(seed ^ 0x63d83595u);
        let uv = (vec2<f32>(id.xy)+vec2<f32>(random(&camera_rng),random(&camera_rng)))/vec2<f32>(params.size.xy);
        let direction = normalize(params.forward.xyz+(2.0*uv.x-1.0)*params.right.xyz+(1.0-2.0*uv.y)*params.up.xyz);
        if ACTIVE_MODE == 2u {
            let a = trace(params.eye.xyz,direction,seed,0u);
            let b = trace(params.eye.xyz,direction,seed,1u);
            if DENOISE_INPUT {fresh_a+=a.radiance;fresh_b+=b.radiance;}
            pixel.a += vec4<f32>(a.radiance,0.0); pixel.b += vec4<f32>(b.radiance,0.0);
            pixel.stats_a += vec4<f32>(a.stats,0.0); pixel.stats_b += vec4<f32>(b.stats,0.0);
            pixel.normals_a += vec4<f32>(a.normal,0.0); pixel.normals_b += vec4<f32>(b.normal,0.0);
        } else {
            let a = trace(params.eye.xyz,direction,seed,ACTIVE_MODE);
            if DENOISE_INPUT {fresh_a+=a.radiance;}
            pixel.a += vec4<f32>(a.radiance,0.0); pixel.stats_a += vec4<f32>(a.stats,0.0); pixel.normals_a += vec4<f32>(a.normal,0.0);
        }
    }
    pixels[index] = pixel;
    if DENOISE_INPUT {
        var fresh=select(fresh_a,mix(fresh_b,fresh_a,params.controls.x),ACTIVE_MODE==2u);
        let count=f32(params.size.z);
        fresh/=max(count,1.0);
        frames[index]=guide_frame(id,fresh,count);
    }
    if (u32(params.controls.w) & 8u) != 0u { display_pixel(id,pixel); }
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
        color = estimate;
        color *= exp2(params.controls.z);
        color = srgb(color/(vec3<f32>(1.0)+color)); // Reinhard, then exact sRGB encoding.
    } else if params.transport.w == 1u { color = mix(vec3<f32>(0.1,0.6,1.0),vec3<f32>(1.0,0.25,0.08),stats.x); }
    else if params.transport.w == 2u { color = vec3<f32>(stats.y/f32(max(1u,params.transport.y-1u)),0.1,1.0-stats.y/f32(max(1u,params.transport.y-1u))); }
    else if params.transport.w == 3u { color = vec3<f32>(stats.z/f32(params.transport.y)); }
    else { color = normals; }
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
