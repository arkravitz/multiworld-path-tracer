// SPPC: persistent camera-query ring, separate progressive photon statistics,
// software query-ball index, and bounded uniform three-query reservoirs.
struct SpState {emitted:u32,round:u32,step:u32,valid:atomic<u32>}
struct SpQuery {sample:NeuralSample,global:vec4<f32>,caustic:vec4<f32>,radii:vec2<f32>,birth:u32,valid:u32,environment:vec4<f32>}
struct SpPending {values:array<atomic<u32>,8>}
struct SpStorage {
    state:SpState,
    weights:array<f32,8515>,ema:array<f32,8515>,m:array<f32,8515>,v:array<f32,8515>,
    gradients:array<f32,1089920>,
    hash_weights:array<f32,SP_HASH_PARAMETERS>,hash_ema:array<f32,SP_HASH_PARAMETERS>,
    hash_m:array<f32,SP_HASH_PARAMETERS>,hash_v:array<f32,SP_HASH_PARAMETERS>,
    hash_grad:array<atomic<u32>,SP_HASH_PARAMETERS>,
    queries:array<SpQuery,SP_QUERIES>,pending:array<SpPending,SP_QUERIES>,
    counts:array<atomic<u32>,SP_QUERIES>,offsets:array<u32,SP_QUERIES>,
    indices:array<u32,SP_QUERIES>,blocks:array<u32,SP_BLOCKS>,
    training:array<NeuralSample,128>,
}
struct SpParams {config:vec4<u32>,radii:vec4<f32>,low:vec4<f32>,scale:vec4<f32>}
@group(1) @binding(0) var<storage,read_write> sp:SpStorage;
@group(1) @binding(1) var<uniform> sp_params:SpParams;
@group(1) @binding(3) var<storage,read_write> neural_queries:array<NeuralSample>;
const CAUSTICS_ENABLED:bool=false;
fn caustic_estimate(world:u32,p:vec3<f32>,n:vec3<f32>,c:vec3<f32>,b:u32,d:vec3<f32>)->vec3<f32>{return vec3<f32>(0.0);}
fn caustic_visible(i:u32,w:u32)->vec3<f32>{return vec3<f32>(0.0);}
fn photon_nrc_ready()->bool {return sp.state.step>=32u && params.size.w>=8u;}
fn neural_resolve_matrix(lane:u32)->vec3<f32>{return vec3<f32>(0.0);}
fn neural_enqueue(index:u32,sample:NeuralSample){neural_queries[index]=sample;}
fn neural_clear_queries(index:u32){
    let slots=select(1u,2u,ACTIVE_MODE==2u);neural_queries[index*slots].label=vec4<f32>(0.0);
    if slots==2u {neural_queries[index*slots+1u].label=vec4<f32>(0.0);}
}
fn sp_cell(p:vec3<f32>)->vec3<i32>{return vec3<i32>(floor(p/max(sp_params.radii.x,sp_params.radii.y)));}
fn sp_bucket(cell:vec3<i32>)->u32{return hash(bitcast<u32>(cell.x)^hash(bitcast<u32>(cell.y))^hash(bitcast<u32>(cell.z)))&(SP_QUERIES-1u);}
fn sp_add(q:u32,c:u32,value:f32){
    var old=atomicLoad(&sp.pending[q].values[c]);
    loop {let result=atomicCompareExchangeWeak(&sp.pending[q].values[c],old,bitcast<u32>(bitcast<f32>(old)+value));if result.exchanged {break;}old=result.old_value;}
}
// Independent camera-sampled environment suffix. Surface-emission photons and
// environment paths have disjoint sources; environment must not disappear.
fn sp_environment(s:NeuralSample,seed:u32)->vec3<f32>{
    var radiance=vec3<f32>(0.0);
    let world=u32(s.position.w);var rng=seed;var ro=s.position.xyz+s.normal.xyz*0.002;
    var rd=cosine_direction(s.normal.xyz,&rng);var weight=s.albedo.xyz;
    var medium_active=false;var color=vec3<f32>(1.0);var ior=1.0;var eta_scale=1.0;
    if (params.lights.w&(1u<<world))!=0u {let h=glass_boundary(world,ro,rd);if h.t<1e29 && h.front<0.5 {medium_active=true;color=h.color;ior=h.metal-2.0;}}
    for(var depth=0u;depth<u32(s.normal.w);depth++){
        let h=intersect(world,ro,rd);if h.t>1e29 {return radiance+weight*vec3<f32>(0.045,0.055,0.075);}
        if medium_active {weight*=pow(clamp(color,vec3<f32>(0.0),vec3<f32>(1.0)),vec3<f32>(h.t));}
        if h.light_pdf==0.0 {radiance+=weight*h.emission;}
        let p=ro+h.t*rd;var transmitted=false;
        if h.metal>=2.0 {
            let ei=select(h.metal-2.0,ior,h.front>0.5);let et=select(1.0,h.metal-2.0,h.front>0.5);
            if random(&rng)<dielectric_fresnel(clamp(-dot(rd,h.normal),0.0,1.0),ei,et){rd=reflect(rd,h.normal);}
            else {rd=refract(rd,h.normal,ei/et);weight*=ei*ei/(et*et);eta_scale*=et*et/(ei*ei);medium_active=h.front>0.5;ior=et;color=h.color;transmitted=true;}
        }else if h.metal>0.5 {rd=reflect(rd,h.normal);weight*=h.color;}
        else {rd=cosine_direction(h.normal,&rng);weight*=h.color;}
        if max(weight.x,max(weight.y,weight.z))<=0.0 {break;}
        ro=p+h.normal*select(0.002,-0.002,transmitted);
        if depth>=3u {let survival=clamp(max(weight.x,max(weight.y,weight.z))*eta_scale,0.05,0.95);if random(&rng)>=survival {break;}weight/=survival;}
    }
    return radiance;
}
@compute @workgroup_size(64)
fn sp_initialize(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x+id.y*65536u;
    if i<8515u {
        let bias=(i>=4096u&&i<4160u)||(i>=8256u&&i<8320u)||i>=8512u;
        let w=select((f32(hash(i^0x7193ab9u)>>8u)/16777216.0-0.5)*0.3,0.0,bias);
        sp.weights[i]=w;sp.ema[i]=w;sp.m[i]=0.0;sp.v[i]=0.0;
    }
    if i<SP_HASH_PARAMETERS {
        let w=(f32(hash(i^0x1f123bb5u)>>8u)/16777216.0-0.5)*0.0002;
        sp.hash_weights[i]=w;sp.hash_ema[i]=w;sp.hash_m[i]=0.0;sp.hash_v[i]=0.0;atomicStore(&sp.hash_grad[i],0u);
    }
    if i<SP_QUERIES {sp.queries[i].valid=0u;}
    if i==0u {sp.state.emitted=0u;sp.state.round=0u;sp.state.step=0u;atomicStore(&sp.state.valid,0u);}
}
@compute @workgroup_size(64)
fn sp_prepare(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;if i>=SP_QUERIES {return;}
    atomicStore(&sp.counts[i],0u);for(var c=0u;c<8u;c++){atomicStore(&sp.pending[i].values[c],0u);}
    if i==0u {atomicStore(&sp.state.valid,0u);}
    let replace=select(sp_params.config.w,SP_QUERIES,sp_params.config.z==0u);
    if i>=replace {return;}
    let index=(sp_params.config.z*sp_params.config.w+i)&(SP_QUERIES-1u);
    var rng=hash(i^hash(sp_params.config.z)^params.transport.z^0xfcb7321u);
    let total=params.size.x*params.size.y*select(1u,2u,params.transport.x==2u);
    var sample:NeuralSample;
    for(var attempt=0u;attempt<8u;attempt++) {sample=neural_queries[min(u32(random(&rng)*f32(total)),total-1u)];if sample.label.w>0.0 {break;}}
    var q:SpQuery;q.sample=sample;
    q.radii=sp_params.radii.xy*sp_params.radii.xy;q.birth=sp.state.emitted;q.valid=select(0u,1u,sample.label.w>0.0);
    if q.valid!=0u {q.environment=vec4<f32>(sp_environment(sample,hash(rng)),1.0);}
    sp.queries[index]=q;
}
var<workgroup> sp_scan:array<u32,256>;
fn sp_prefix(lane:u32){
    workgroupBarrier();for(var stride=1u;stride<256u;stride*=2u){var add=0u;if lane>=stride {add=sp_scan[lane-stride];}workgroupBarrier();sp_scan[lane]+=add;workgroupBarrier();}
}
@compute @workgroup_size(64)
fn sp_count(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x<SP_QUERIES && sp.queries[id.x].valid!=0u {atomicAdd(&sp.counts[sp_bucket(sp_cell(sp.queries[id.x].sample.position.xyz))],1u);}
}
@compute @workgroup_size(256)
fn sp_scan_buckets(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>){
    let count=atomicLoad(&sp.counts[id.x]);sp_scan[lane]=count;sp_prefix(lane);sp.offsets[id.x]=sp_scan[lane]-count;
    if lane==255u {sp.blocks[g.x]=sp_scan[lane];}atomicStore(&sp.counts[id.x],0u);
}
@compute @workgroup_size(256)
fn sp_scan_blocks(@builtin(local_invocation_index) lane:u32){var n=0u;if lane<SP_BLOCKS {n=sp.blocks[lane];}sp_scan[lane]=n;sp_prefix(lane);if lane<SP_BLOCKS {sp.blocks[lane]=sp_scan[lane]-n;}}
@compute @workgroup_size(64)
fn sp_scatter(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x>=SP_QUERIES || sp.queries[id.x].valid==0u {return;}
    let b=sp_bucket(sp_cell(sp.queries[id.x].sample.position.xyz));let start=sp.offsets[b]+sp.blocks[b/256u];let i=atomicAdd(&sp.counts[b],1u);sp.indices[start+i]=id.x;
}
fn sp_collect(world:u32,p:vec3<f32>,n:vec3<f32>,flux:vec3<f32>,segments:u32,caustic:bool,rng:ptr<function,u32>){
    var power=flux;
    if !caustic {if random(rng)>=sp_params.radii.z {return;}power/=sp_params.radii.z;}
    let cell=sp_cell(p);var selected:array<u32,3>;var seen=0u;
    for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
        let neighbor=cell+vec3<i32>(x,y,z);let bucket=sp_bucket(neighbor);let start=sp.offsets[bucket]+sp.blocks[bucket/256u];let count=atomicLoad(&sp.counts[bucket]);
        for(var i=0u;i<count;i++){
            let index=sp.indices[start+i];let q=sp.queries[index];
            if u32(q.sample.position.w)!=world || !all(sp_cell(q.sample.position.xyz)==neighbor) || segments>u32(q.sample.normal.w) || dot(n,q.sample.normal.xyz)<0.5 {continue;}
            let delta=p-q.sample.position.xyz;let radius2=select(q.radii.x,q.radii.y,caustic);
            if dot(delta,delta)>radius2 {continue;}
            seen++;if seen<=3u {selected[seen-1u]=index;}else{let slot=min(u32(random(rng)*f32(seen)),seen-1u);if slot<3u {selected[slot]=index;}}
        }
    }}}
    let kept=min(seen,3u);if kept==0u {return;}
    let factor=f32(seen)/f32(kept);let channel=select(0u,4u,caustic);
    for(var i=0u;i<kept;i++){
        let q=selected[i];let outgoing=power*sp.queries[q].sample.albedo.xyz*(factor/PI);
        for(var c=0u;c<3u;c++){sp_add(q,channel+c,outgoing[c]);}atomicAdd(&sp.pending[q].values[channel+3u],1u);
    }
}
@compute @workgroup_size(64)
fn sp_trace_photons(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x>=sp_params.config.x {return;}
    let world=id.y;if (params.transport.x==0u&&world==1u)||(params.transport.x==1u&&world==0u){return;}
    let count=select(params.lights.x,params.lights.y,world==1u);if count==0u {return;}
    var rng=hash(id.x^hash(sp_params.config.z)^hash(world+67u)^params.transport.z^0xadd78f1u);
    let offset=select(params.acceleration.x,params.acceleration.y,world==1u);let u=random(&rng);var low=0u;var high=count-1u;
    loop {if low>=high {break;}let mid=(low+high)/2u;if u<node_at(world,offset+mid).low.x {high=mid;}else{low=mid+1u;}}
    let light=object_at(world,node_at(world,offset+low).object);var ro:vec3<f32>;var n:vec3<f32>;var area:f32;
    if light.extent.w<0.5 {n=oriented_direction(vec3<f32>(0.0,1.0,0.0),1.0-2.0*random(&rng),2.0*PI*random(&rng));ro=light.center.xyz+n*light.center.w;area=4.0*PI*light.center.w*light.center.w;}
    else {
        let areas=8.0*light.extent.yzx*light.extent.zxy;area=areas.x+areas.y+areas.z;var face=random(&rng)*area;var axis=0u;
        if face>=areas.x {face-=areas.x;axis=1u;if face>=areas.y {face-=areas.y;axis=2u;}}
        n=vec3<f32>(0.0);n[axis]=select(-1.0,1.0,face>=areas[axis]*0.5);ro=light.center.xyz;ro[axis]+=n[axis]*light.extent[axis];
        ro[(axis+1u)%3u]+=(2.0*random(&rng)-1.0)*light.extent[(axis+1u)%3u];ro[(axis+2u)%3u]+=(2.0*random(&rng)-1.0)*light.extent[(axis+2u)%3u];
    }
    var rd=cosine_direction(n*select(-1.0,1.0,random(&rng)<0.5),&rng);var flux=light.emission.xyz*(area*2.0*PI/light.emission.w);
    ro+=n*select(-0.002,0.002,dot(rd,n)>0.0);var medium_active=false;var color=vec3<f32>(1.0);var ior=1.0;var last_delta=false;
    if (params.lights.w&(1u<<world))!=0u {let h=glass_boundary(world,ro,rd);if h.t<1e29&&h.front<0.5 {medium_active=true;color=h.color;ior=h.metal-2.0;}}
    for(var depth=0u;depth<params.transport.y;depth++){
        let h=intersect(world,ro,rd);if h.t>1e29 {break;}if medium_active {flux*=pow(clamp(color,vec3<f32>(0.0),vec3<f32>(1.0)),vec3<f32>(h.t));}
        let p=ro+h.t*rd;var transmitted=false;
        if h.metal<=0.5 {
            sp_collect(world,p,h.normal,flux,depth+1u,last_delta,&rng);flux*=h.color;rd=cosine_direction(h.normal,&rng);last_delta=false;
        }else if h.metal>=2.0 {
            let ei=select(h.metal-2.0,ior,h.front>0.5);let et=select(1.0,h.metal-2.0,h.front>0.5);
            if random(&rng)<dielectric_fresnel(clamp(-dot(rd,h.normal),0.0,1.0),ei,et){rd=reflect(rd,h.normal);}
            else{rd=refract(rd,h.normal,ei/et);medium_active=h.front>0.5;ior=et;color=h.color;transmitted=true;}last_delta=true;
        }else{rd=reflect(rd,h.normal);flux*=h.color;last_delta=true;}
        if max(flux.x,max(flux.y,flux.z))<=0.0 {break;}ro=p+h.normal*select(0.002,-0.002,transmitted);
        if depth>=3u {let survival=0.8;if random(&rng)>=survival {break;}flux/=survival;}
    }
}
@compute @workgroup_size(64)
fn sp_finalize(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x>=SP_QUERIES {return;}var q=sp.queries[id.x];if q.valid==0u {return;}
    for(var kind=0u;kind<2u;kind++){
        let base=kind*4u;let m=f32(atomicLoad(&sp.pending[id.x].values[base+3u]));
        var old=select(q.global,q.caustic,kind==1u);
        if m>0.0 {
            let next=old.w+sp_params.radii.w*m;let ratio=next/(old.w+m);
            let added=vec3<f32>(bitcast<f32>(atomicLoad(&sp.pending[id.x].values[base])),bitcast<f32>(atomicLoad(&sp.pending[id.x].values[base+1u])),bitcast<f32>(atomicLoad(&sp.pending[id.x].values[base+2u])));
            old=vec4<f32>((old.xyz+added)*ratio,next);q.radii[kind]*=ratio;
        }
        if kind==0u {q.global=old;}else{q.caustic=old;}
    }
    sp.queries[id.x]=q;
}
@compute @workgroup_size(64)
fn sp_environment_update(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x>=min(SP_QUERIES,1024u){return;}
    let i=(sp_params.config.z*1024u+id.x)&(SP_QUERIES-1u);var q=sp.queries[i];if q.valid==0u {return;}
    q.environment+=vec4<f32>(sp_environment(q.sample,hash(i^hash(sp_params.config.z)^params.transport.z)),1.0);sp.queries[i]=q;
}
@compute @workgroup_size(64)
fn sp_labels(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x>=128u {return;}
    let i=hash(id.x^hash(sp_params.config.z)^0x72187fau)&(SP_QUERIES-1u);let q=sp.queries[i];var s=q.sample;s.label=vec4<f32>(0.0);
    let emitted=sp.state.emitted+sp_params.config.x-q.birth;
    if q.valid!=0u && emitted>=sp_params.config.x*2u {
        let finite=(q.global.xyz/max(q.radii.x,1e-12)+q.caustic.xyz/max(q.radii.y,1e-12))/(PI*f32(emitted));
        let environment=q.environment.xyz/max(1.0,q.environment.w);
        s.label=vec4<f32>(clamp((finite+environment)/(s.albedo.xyz+vec3<f32>(0.01)),vec3<f32>(0.0),vec3<f32>(1000.0)),1.0);atomicAdd(&sp.state.valid,1u);
    }
    sp.training[id.x]=s;
}
