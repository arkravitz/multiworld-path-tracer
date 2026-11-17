// Light-side importance transport. One packed allocation keeps Metal within
// baseline storage-buffer limits. No float-atomic extension is needed.
const CAUSTICS_ENABLED: bool = true;
struct CausticParams {
    targets:array<vec4<f32>,32>, // up to 16 caster bounds per world
    offsets:vec4<u32>, // photons/world, buckets/world, record offset, splat offset (words)
    shape:vec4<f32>, // gather radius, caster counts A/B, reserved
    spheres:array<vec4<f32>,16>, // four (center/radius, IOR) pairs per world
    learning:vec4<u32>, // frozen weights, contribution statistics, focus, reservoirs
    features:vec4<u32>, // methods bitmask, active launch count, round, sphere counts
}
@group(1) @binding(0) var<storage,read_write> photon_data:array<atomic<u32>>;
@group(1) @binding(1) var<uniform> caustics:CausticParams;
fn photon_float(i:u32)->f32 {return bitcast<f32>(atomicLoad(&photon_data[i]));}
fn photon_vec(i:u32)->vec3<f32> {return vec3<f32>(photon_float(i),photon_float(i+1u),photon_float(i+2u));}
fn photon_write(i:u32,v:vec3<f32>) {
    atomicStore(&photon_data[i],bitcast<u32>(v.x));atomicStore(&photon_data[i+1u],bitcast<u32>(v.y));atomicStore(&photon_data[i+2u],bitcast<u32>(v.z));
}
fn photon_add(i:u32,v:f32) {
    var old=atomicLoad(&photon_data[i]);
    loop {let next=bitcast<u32>(bitcast<f32>(old)+v);let result=atomicCompareExchangeWeak(&photon_data[i],old,next);if result.exchanged {break;}old=result.old_value;}
}
fn photon_cell(p:vec3<f32>)->vec3<i32> {return vec3<i32>(floor(p/caustics.shape.x));}
fn photon_bucket(cell:vec3<i32>,world:u32)->u32 {
    return world*caustics.offsets.y+(hash(bitcast<u32>(cell.x)^hash(bitcast<u32>(cell.y))^hash(bitcast<u32>(cell.z)))&(caustics.offsets.y-1u));
}
// Splat power to a pinhole pixel using the receiver-area / pixel-solid-angle
// Jacobian. Binning photon positions directly avoids a display-space blur.
fn photon_splat(world:u32,p:vec3<f32>,n:vec3<f32>,color:vec3<f32>,flux:vec3<f32>,length:u32)->f32 {
    if length>params.transport.y {return 0.0;}
    let delta=p-params.eye.xyz;let z=dot(delta,params.forward.xyz);
    if z<=0.0 {return 0.0;}
    let x=dot(delta,params.right.xyz)/(z*dot(params.right.xyz,params.right.xyz));
    let y=dot(delta,params.up.xyz)/(z*dot(params.up.xyz,params.up.xyz));
    let uv=vec2<f32>(0.5+0.5*x,0.5-0.5*y);
    if any(uv<vec2<f32>(0.0)) || any(uv>=vec2<f32>(1.0)) {return 0.0;}
    let d2=dot(delta,delta);let direction=delta/sqrt(d2);let cosine=dot(n,-direction);
    if cosine<=0.0 {return 0.0;}
    if shadow_occluded(world,p+n*0.002,-direction,sqrt(d2)-0.004) {return 0.0;}
    let solid_angle=4.0*length_vector(params.right.xyz)*length_vector(params.up.xyz)*pow(z/sqrt(d2),3.0)/f32(params.size.x*params.size.y);
    let value=flux*color*(cosine/(PI*d2*solid_angle));
    let pixel=vec2<u32>(uv*vec2<f32>(params.size.xy));
    let index=caustics.offsets.w+(world*params.size.x*params.size.y+pixel.y*params.size.x+pixel.x)*3u;
    photon_add(index,value.x);photon_add(index+1u,value.y);photon_add(index+2u,value.z);
    return dot(value,vec3<f32>(0.2126,0.7152,0.0722));
}
// Keep the parameter named 'length' above without hiding the builtin.
fn length_vector(v:vec3<f32>)->f32 {return sqrt(dot(v,v));}
fn caustic_estimate(world:u32,p:vec3<f32>,n:vec3<f32>,color:vec3<f32>,bounce:u32,camera_direction:vec3<f32>)->vec3<f32> {
    if bounce==0u {return vec3<f32>(0.0);}
    let cell=photon_cell(p);let radius=caustics.shape.x;var flux=vec3<f32>(0.0);
    for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
        let neighbor=cell+vec3<i32>(x,y,z);let bucket=photon_bucket(neighbor,world);
        let start=atomicLoad(&photon_data[photon_offsets()+bucket])+atomicLoad(&photon_data[photon_blocks()+bucket/256u]);
        let count=atomicLoad(&photon_data[bucket]);
        for(var i=0u;i<count;i++) {
            let id=atomicLoad(&photon_data[photon_indices()+start+i]);
            let base=caustics.offsets.z+id*16u;let position=photon_vec(base);
            // A hash collision must never count a photon again in another cell.
            if all(photon_cell(position)==neighbor) && bounce+atomicLoad(&photon_data[base+9u])<=params.transport.y {
                let delta=position-p;let plane=dot(delta,n);let d=length_vector(delta);
                if d<radius && abs(plane)<radius*0.1 && dot(n,photon_vec(base+3u))>0.9 {
                    let contribution=photon_vec(base+6u)*(3.0*(1.0-d/radius)/(PI*radius*radius));
                    flux+=contribution;
                    if (caustics.features.x&1u)!=0u {
                        let cell=guide_cell(world,atomicLoad(&photon_data[base+12u]),atomicLoad(&photon_data[base+13u]));
                        photon_add(caustics.learning.y+cell*2u+1u,min(1000.0,max(0.0,dot(contribution,vec3<f32>(0.2126,0.7152,0.0722)))));
                    }
                }
            }
        }
    }}}
    return color*flux/PI;
}
@compute @workgroup_size(64)
fn photon_clear(@builtin(global_invocation_id) id:vec3<u32>) {
    let index=id.x+id.y*65536u;
    if index<caustics.offsets.x*2u {atomicStore(&photon_data[caustics.offsets.z+index*16u+11u],0u);}
    if index<caustics.offsets.y*2u {atomicStore(&photon_data[index],0u);}
    if index<params.size.x*params.size.y*6u {atomicStore(&photon_data[caustics.offsets.w+index],0u);}
}
struct LightSample {
    position:vec3<f32>, normal:vec3<f32>, flux:vec3<f32>, color:vec3<f32>,
    depth:u32, valid:u32, light:u32, caster:u32,
}
fn trace_light_sample(world:u32,seed:u32)->LightSample {
    var result:LightSample;
    let count=select(params.lights.x,params.lights.y,world==1u);if count==0u {return result;}
    var rng=seed;
    let offset=select(params.acceleration.x,params.acceleration.y,world==1u);
    let low=sample_photon_light(world,count,&rng);
    result.light=low;
    let light=object_at(world,node_at(world,offset+low).object);
    var ro:vec3<f32>;var normal:vec3<f32>;var area:f32;
    if light.extent.w<0.5 {
        normal=oriented_direction(vec3<f32>(0.0,1.0,0.0),1.0-2.0*random(&rng),2.0*PI*random(&rng));
        ro=light.center.xyz+normal*light.center.w;area=4.0*PI*light.center.w*light.center.w;
    }else{
        let areas=8.0*light.extent.yzx*light.extent.zxy;area=areas.x+areas.y+areas.z;
        var face=random(&rng)*area;var axis=0u;
        if face>=areas.x {face-=areas.x;axis=1u;if face>=areas.y {face-=areas.y;axis=2u;}}
        normal=vec3<f32>(0.0);normal[axis]=select(-1.0,1.0,face>=areas[axis]*0.5);
        ro=light.center.xyz;ro[axis]+=normal[axis]*light.extent[axis];
        let a=(axis+1u)%3u;let b=(axis+2u)%3u;
        ro[a]+=(2.0*random(&rng)-1.0)*light.extent[a];ro[b]+=(2.0*random(&rng)-1.0)*light.extent[b];
    }
    let target_count=u32(select(caustics.shape.y,caustics.shape.z,world==1u));
    let chosen=sample_photon_caster(world,low,target_count,&rng);
    result.caster=chosen;
    let proposal=emission_direction(world,low,target_count,chosen,ro,normal,&rng);
    var rd=proposal.xyz;let pdf=proposal.w;
    let cosine=abs(dot(rd,normal));if cosine<=0.0 || pdf<=0.0 {return result;}
    var flux=light.emission.xyz*(area*cosine/(photon_light_pdf(world,low,count)*pdf));
    ro+=normal*select(-0.002,0.002,dot(rd,normal)>0.0);
    var medium_active=false;var medium_color=vec3<f32>(1.0);var medium_ior=1.0;
    var may_start_inside=false;
    for(var i=0u;i<target_count;i++) {let bound=caustics.targets[world*16u+i];if dot(ro-bound.xyz,ro-bound.xyz)<=bound.w*bound.w {may_start_inside=true;}}
    if may_start_inside && (params.lights.w&(1u<<world))!=0u {let boundary=glass_boundary(world,ro,rd);if boundary.t<1e29 && boundary.front<0.5 {medium_active=true;medium_color=boundary.color;medium_ior=boundary.metal-2.0;}}
    for(var depth=0u;depth+2u<=params.transport.y;depth++){
        let hit=intersect(world,ro,rd);if hit.t>1e29 {return result;}
        if medium_active {flux*=pow(clamp(medium_color,vec3<f32>(0.0),vec3<f32>(1.0)),vec3<f32>(hit.t));}
        let p=ro+hit.t*rd;
        if hit.metal<=0.5 {
            if depth==0u {return result;} // Direct lighting remains exclusively camera MIS.
            result.position=p;result.normal=hit.normal;result.flux=flux;
            result.color=hit.color;result.depth=depth+2u;result.valid=1u;
            return result;
        }
        var transmitted=false;
        if hit.metal>=2.0 {
            let eta_i=select(hit.metal-2.0,medium_ior,hit.front>0.5);let eta_t=select(1.0,hit.metal-2.0,hit.front>0.5);
            let cosine_in=clamp(-dot(rd,hit.normal),0.0,1.0);let fresnel=dielectric_fresnel(cosine_in,eta_i,eta_t);
            if random(&rng)<fresnel {rd=reflect(rd,hit.normal);}else{
                let eta=eta_i/eta_t;rd=normalize(eta*rd+(eta*cosine_in-sqrt(max(0.0,1.0-eta*eta*(1.0-cosine_in*cosine_in))))*hit.normal);
                // Importance transport carries flux: NO radiance eta^2 factor.
                medium_ior=eta_t;medium_active=hit.front>0.5;medium_color=hit.color;transmitted=true;
            }
        }else{rd=reflect(rd,hit.normal);flux*=hit.color;}
        ro=p+select(0.002,-0.002,transmitted)*hit.normal;
    }
    return result;
}

fn caustic_visible(index:u32,world:u32)->vec3<f32> {
    return photon_vec(caustics.offsets.w+(world*params.size.x*params.size.y+index)*3u);
}

fn caster_cone(caster:vec4<f32>,ro:vec3<f32>)->f32 {
    let d2=dot(caster.xyz-ro,caster.xyz-ro);
    if d2<=caster.w*caster.w {return 2.0;}
    let ratio=min(1.0,caster.w*caster.w/d2);return ratio/(1.0+sqrt(1.0-ratio));
}

// Counting-sort the hash grid. The 256-bucket local scans and one block scan
// replace linked-list pointer chasing with contiguous ranges, without a cap.
fn photon_offsets()->u32 {return caustics.offsets.y*2u;}
fn photon_blocks()->u32 {return caustics.offsets.y*4u;}
fn photon_indices()->u32 {return caustics.offsets.z+caustics.offsets.x*32u;}
var<workgroup> photon_scan:array<u32,256>;
fn photon_inclusive_scan(lane:u32) {
    workgroupBarrier();
    for(var stride=1u;stride<256u;stride*=2u){
        var add=0u;if lane>=stride {add=photon_scan[lane-stride];}
        workgroupBarrier();photon_scan[lane]+=add;workgroupBarrier();
    }
}
@compute @workgroup_size(256)
fn photon_scan_buckets(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>) {
    let count=atomicLoad(&photon_data[id.x]);photon_scan[lane]=count;photon_inclusive_scan(lane);
    atomicStore(&photon_data[photon_offsets()+id.x],photon_scan[lane]-count);
    if lane==255u {atomicStore(&photon_data[photon_blocks()+group.x],photon_scan[lane]);}
    atomicStore(&photon_data[id.x],0u);
}
@compute @workgroup_size(256)
fn photon_scan_blocks(@builtin(local_invocation_index) lane:u32) {
    var count=0u;if lane<caustics.offsets.y*2u/256u {count=atomicLoad(&photon_data[photon_blocks()+lane]);}
    photon_scan[lane]=count;photon_inclusive_scan(lane);
    atomicStore(&photon_data[photon_blocks()+lane],photon_scan[lane]-count);
}
@compute @workgroup_size(64)
fn photon_scatter(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=caustics.offsets.x*2u {return;}
    let record=caustics.offsets.z+id.x*16u;if atomicLoad(&photon_data[record+11u])==0u {return;}
    let bucket=atomicLoad(&photon_data[record+10u]);
    let start=atomicLoad(&photon_data[photon_offsets()+bucket])+atomicLoad(&photon_data[photon_blocks()+bucket/256u]);
    let index=atomicAdd(&photon_data[bucket],1u);
    atomicStore(&photon_data[photon_indices()+start+index],id.x);
}

fn photon_nrc_ready()->bool {return false;}
