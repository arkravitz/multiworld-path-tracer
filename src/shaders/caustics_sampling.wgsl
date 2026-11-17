override PHOTON_METHODS:u32=0u;
// Frozen eight-round learning epochs keep replayed primary-sample mappings
// independent of the reservoirs reused within each epoch.
fn guide_cell(world:u32,light:u32,caster:u32)->u32 {return world*512u+min(light,31u)*16u+min(caster,15u);}
fn guide_value(world:u32,light:u32,caster:u32)->f32 {
    return max(1e-6,photon_float(caustics.learning.x+guide_cell(world,light,caster)));
}
fn caster_probability(world:u32,light:u32,caster:u32,count:u32)->f32 {
    if count==0u {return 1.0;}
    if (PHOTON_METHODS&1u)==0u || light>=32u || select(params.lights.x,params.lights.y,world==1u)>32u {return 1.0/f32(count);}
    return photon_float(caustics.learning.y-1088u+guide_cell(world,light,caster));
}
fn light_utility(world:u32,light:u32)->f32 {
    var total=0.0;for(var i=0u;i<16u;i++){total+=guide_value(world,light,i);}return total;
}
fn photon_light_pdf(world:u32,light:u32,count:u32)->f32 {
    let offset=select(params.acceleration.x,params.acceleration.y,world==1u);
    let base=object_at(world,node_at(world,offset+light).object).emission.w;
    if (PHOTON_METHODS&1u)==0u || count>32u {return base;}
    return photon_float(caustics.learning.y-64u+world*32u+light);
}
@compute @workgroup_size(64)
fn photon_probabilities(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=64u || caustics.features.z%8u!=0u || (caustics.features.x&1u)==0u {return;}
    let world=id.x/32u;let light=id.x%32u;
    let count=select(params.lights.x,params.lights.y,world==1u);
    if light>=count || count>32u {return;}
    let offset=select(params.acceleration.x,params.acceleration.y,world==1u);
    let base=object_at(world,node_at(world,offset+light).object).emission.w;
    var total=0.0;for(var i=0u;i<count;i++){total+=light_utility(world,i);}
    let probability=0.2*base+0.8*light_utility(world,light)/max(total,1e-12);
    atomicStore(&photon_data[caustics.learning.y-64u+id.x],bitcast<u32>(probability));
    let casters=u32(select(caustics.shape.y,caustics.shape.z,world==1u));
    var sum=0.0;for(var i=0u;i<casters;i++){sum+=guide_value(world,light,i);}
    for(var i=0u;i<casters;i++) {
        let p=0.2/f32(casters)+0.8*guide_value(world,light,i)/max(1e-12,sum);
        atomicStore(&photon_data[caustics.learning.y-1088u+guide_cell(world,light,i)],bitcast<u32>(p));
    }
}
fn sample_photon_light(world:u32,count:u32,rng:ptr<function,u32>)->u32 {
    let u=random(rng);var cdf=0.0;
    for(var i=0u;i<count;i++){cdf+=photon_light_pdf(world,i,count);if u<cdf {return i;}}
    return count-1u;
}
fn sample_photon_caster(world:u32,light:u32,count:u32,rng:ptr<function,u32>)->u32 {
    let u=random(rng);var cdf=0.0;
    for(var i=0u;i<count;i++){cdf+=caster_probability(world,light,i,count);if u<cdf {return i;}}
    return max(1u,count)-1u;
}
fn caustic_basis(n:vec3<f32>)->mat2x3<f32> {
    let helper=select(vec3<f32>(0.0,1.0,0.0),vec3<f32>(1.0,0.0,0.0),abs(n.y)>0.9);
    let t=normalize(cross(helper,n));return mat2x3<f32>(t,cross(n,t));
}
fn manifold_sphere_t(ro:vec3<f32>,rd:vec3<f32>,sphere:vec4<f32>)->f32 {
    let oc=ro-sphere.xyz;let b=dot(oc,rd);let disc=b*b-dot(oc,oc)+sphere.w*sphere.w;
    if disc<0.0 {return -1.0;}
    let a=-b-sqrt(disc);if a>1e-5 {return a;}
    let z=-b+sqrt(disc);return select(-1.0,z,z>1e-5);
}
// Shooting through exactly two interfaces of one smooth sphere. The final
// ordinary path tracer remains responsible for visibility and actual scattering.
fn manifold_landing(ro:vec3<f32>,rd:vec3<f32>,sphere:vec4<f32>,ior:f32,p:vec3<f32>,n:vec3<f32>)->vec4<f32> {
    let t=manifold_sphere_t(ro,rd,sphere);if t<0.0 {return vec4<f32>(0.0);}
    let enter=ro+t*rd;let en=normalize(enter-sphere.xyz);
    let inside=refract(rd,en,1.0/ior);if dot(inside,inside)<0.5 {return vec4<f32>(0.0);}
    let origin=enter-en*0.0001;let exit_t=manifold_sphere_t(origin,inside,sphere);
    if exit_t<0.0 {return vec4<f32>(0.0);}
    let exit_point=origin+exit_t*inside;let exit_n=normalize(exit_point-sphere.xyz);
    let outgoing=refract(inside,-exit_n,ior);if dot(outgoing,outgoing)<0.5 {return vec4<f32>(0.0);}
    let denominator=dot(outgoing,n);if abs(denominator)<1e-6 {return vec4<f32>(0.0);}
    let distance=dot(p-exit_point,n)/denominator;if distance<=0.0 {return vec4<f32>(0.0);}
    return vec4<f32>(exit_point+distance*outgoing,1.0);
}
fn manifold_proposal(world:u32,ro:vec3<f32>,sphere_index:u32)->vec4<f32> {
    let focus=caustics.learning.z+world*8u;let normal=photon_vec(focus+4u);
    if photon_float(focus+7u)<0.5 {return vec4<f32>(0.0);}
    atomicAdd(&photon_data[caustics.learning.x-8u+world*4u],1u);
    let p=photon_vec(focus);let sphere=caustics.spheres[world*8u+sphere_index*2u];
    let ior=caustics.spheres[world*8u+sphere_index*2u+1u].x;
    if length_vector(ro-sphere.xyz)<=sphere.w*1.001 {return vec4<f32>(0.0);}
    let axis=normalize(sphere.xyz-ro);let basis=caustic_basis(axis);let plane=caustic_basis(normal);
    var uv=vec2<f32>(0.0);
    for(var step=0u;step<8u;step++) {
        let rd=normalize(axis+basis*uv);let hit=manifold_landing(ro,rd,sphere,ior,p,normal);
        if hit.w==0.0 {return vec4<f32>(0.0);}
        let residual=vec2<f32>(dot(hit.xyz-p,plane[0]),dot(hit.xyz-p,plane[1]));
        if length(residual)<max(0.0002,sphere.w*0.002) {
            // Finite cone, never a zero-measure delta ray. Its exact directional
            // PDF is mixed with the full-support baseline below.
            let cone=clamp(0.5*pow(max(0.03,sphere.w*0.08)/max(0.1,length_vector(p-ro)),2.0),0.00002,0.005);
            atomicAdd(&photon_data[caustics.learning.x-7u+world*4u],1u);
            return vec4<f32>(rd,cone);
        }
        let epsilon=0.0005;
        let hx=manifold_landing(ro,normalize(axis+basis*(uv+vec2<f32>(epsilon,0.0))),sphere,ior,p,normal);
        let hy=manifold_landing(ro,normalize(axis+basis*(uv+vec2<f32>(0.0,epsilon))),sphere,ior,p,normal);
        if hx.w==0.0 || hy.w==0.0 {return vec4<f32>(0.0);}
        let dx=(hx.xyz-hit.xyz)/epsilon;let dy=(hy.xyz-hit.xyz)/epsilon;
        let a=vec2<f32>(dot(dx,plane[0]),dot(dx,plane[1]));
        let b=vec2<f32>(dot(dy,plane[0]),dot(dy,plane[1]));let det=a.x*b.y-a.y*b.x;
        if abs(det)<1e-7 {return vec4<f32>(0.0);}
        let update=vec2<f32>(b.y*residual.x-b.x*residual.y,-a.y*residual.x+a.x*residual.y)/det;
        uv-=update*min(1.0,0.15/max(length(update),1e-8));
    }
    return vec4<f32>(0.0);
}
fn base_direction_pdf(world:u32,light:u32,count:u32,ro:vec3<f32>,normal:vec3<f32>,rd:vec3<f32>)->f32 {
    var pdf=abs(dot(rd,normal))/(2.0*PI);if count==0u {return pdf;}pdf*=0.1;
    for(var i=0u;i<count;i++) {
        let caster=caustics.targets[world*16u+i];let delta=caster.xyz-ro;
        let axis=normalize(select(vec3<f32>(0.0,1.0,0.0),delta,dot(delta,delta)>1e-12));
        let cone=caster_cone(caster,ro);
        if dot(rd,axis)>=1.0-cone {pdf+=0.9*caster_probability(world,light,i,count)/(2.0*PI*cone);}
    }
    return pdf;
}
fn emission_direction(world:u32,light:u32,count:u32,chosen:u32,ro:vec3<f32>,normal:vec3<f32>,rng:ptr<function,u32>)->vec4<f32> {
    // Choose the sphere BEFORE drawing any direction. Conditional full-support
    // mixtures are normalized even when the Newton solve has multiple roots.
    var manifold=vec4<f32>(0.0);let spheres=(caustics.features.w>>(world*8u))&255u;
    if (PHOTON_METHODS&2u)!=0u && spheres>0u {
        manifold=manifold_proposal(world,ro,min(u32(random(rng)*f32(spheres)),spheres-1u));
    }
    var rd=cosine_direction(normal*select(-1.0,1.0,random(rng)<0.5),rng);
    if count>0u && random(rng)<0.9 {
        let caster=caustics.targets[world*16u+chosen];let delta=caster.xyz-ro;
        let axis=normalize(select(vec3<f32>(0.0,1.0,0.0),delta,dot(delta,delta)>1e-12));
        rd=oriented_direction(axis,1.0-random(rng)*caster_cone(caster,ro),2.0*PI*random(rng));
    }
    if manifold.w>0.0 && random(rng)<0.5 {rd=oriented_direction(manifold.xyz,1.0-random(rng)*manifold.w,2.0*PI*random(rng));}
    var pdf=base_direction_pdf(world,light,count,ro,normal,rd);
    if manifold.w>0.0 {
        pdf*=0.5;if dot(rd,manifold.xyz)>=1.0-manifold.w {pdf+=0.5/(2.0*PI*manifold.w);}
    }
    return vec4<f32>(rd,pdf);
}
// Positive target gives every seed support, including currently missed paths.
fn photon_target(sample:LightSample)->f32 {
    let delta=params.eye.xyz-sample.position;
    let direction=delta*inverseSqrt(max(1e-10,dot(delta,delta)));let d2=max(0.1,dot(delta,delta));
    let z=-dot(delta,params.forward.xyz);var visible=0.0;
    if z>0.0 {
        let x=-dot(delta,params.right.xyz)/(z*dot(params.right.xyz,params.right.xyz));
        let y=-dot(delta,params.up.xyz)/(z*dot(params.up.xyz,params.up.xyz));
        if abs(x)<1.0 && abs(y)<1.0 {visible=max(0.0,dot(sample.normal,direction))*10.0/d2;}
    }
    // Keep power support for receivers reached through glass, while favoring
    // first-diffuse endpoints that can contribute to the pinhole image.
    return 0.001+sqrt(max(0.0,dot(sample.flux,vec3<f32>(0.2126,0.7152,0.0722))))*(0.1+visible);
}
fn photon_feedback(sample:LightSample) {
    if (PHOTON_METHODS&1u)==0u || sample.light>=32u {return;}
    let world=feedback_world; // set by the invocation, no cross-world statistics
    let cell=guide_cell(world,sample.light,sample.caster);
    atomicAdd(&photon_data[caustics.learning.y+cell*2u],1u);
    if sample.valid!=0u {
        photon_add(caustics.learning.y+cell*2u+1u,min(100000.0,photon_target(sample)-0.001));
    }
}
var<private> feedback_world:u32;
@compute @workgroup_size(64)
fn photon_learn(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=1024u || caustics.features.z%8u!=0u {return;}
    if id.x<8u && caustics.features.z==0u {atomicStore(&photon_data[caustics.learning.x-8u+id.x],0u);}
    let stats=caustics.learning.y+id.x*2u;
    if caustics.features.z==0u {
        atomicStore(&photon_data[caustics.learning.x+id.x],bitcast<u32>(1.0));
    }else{
        let count=f32(atomicLoad(&photon_data[stats]));
        if count>0.0 {
            let previous=photon_float(caustics.learning.x+id.x);
            let observed=photon_float(stats+1u)/count;
            atomicStore(&photon_data[caustics.learning.x+id.x],bitcast<u32>(max(1e-6,mix(previous,observed,0.5))));
        }
    }
    atomicStore(&photon_data[stats],0u);atomicStore(&photon_data[stats+1u],0u);
}
var<workgroup> focus_score:array<f32,256>;
var<workgroup> focus_id:array<u32,256>;
@compute @workgroup_size(256)
fn photon_focus(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>) {
    if caustics.features.z%8u!=0u || (caustics.features.x&2u)==0u {return;}
    let world=group.x;let focus=caustics.learning.z+world*8u;
    if caustics.features.z==0u {if lane==0u {atomicStore(&photon_data[focus+7u],0u);}return;}
    var best=-1e30;var selected=0xffffffffu;
    for(var i=lane;i<caustics.features.y;i+=256u) {
        let base=caustics.offsets.z+(world*caustics.offsets.x+i)*16u;
        if atomicLoad(&photon_data[base+11u])==0u {continue;}
        let p=photon_vec(base);let n=photon_vec(base+3u);let delta=params.eye.xyz-p;
        let benefit=max(0.05,dot(n,normalize(delta)))/max(1.0,dot(delta,delta));
        let u=max(1e-7,f32(hash(i^hash(caustics.features.z+world*731u))>>8u)/16777216.0);
        let score=log(u)/benefit;
        if score>best {best=score;selected=base;}
    }
    focus_score[lane]=best;focus_id[lane]=selected;workgroupBarrier();
    for(var stride=128u;stride>0u;stride/=2u) {
        if lane<stride && focus_score[lane+stride]>focus_score[lane] {
            focus_score[lane]=focus_score[lane+stride];focus_id[lane]=focus_id[lane+stride];
        }workgroupBarrier();
    }
    if lane==0u && focus_id[0]!=0xffffffffu {
        photon_write(focus,photon_vec(focus_id[0]));photon_write(focus+4u,photon_vec(focus_id[0]+3u));
        atomicStore(&photon_data[focus+7u],bitcast<u32>(1.0));
    }
}
fn read_reservoir_sample(base:u32)->LightSample {
    var s:LightSample;s.position=photon_vec(base);s.depth=atomicLoad(&photon_data[base+3u]);
    s.normal=photon_vec(base+4u);s.valid=atomicLoad(&photon_data[base+7u]);
    s.flux=photon_vec(base+8u);s.light=atomicLoad(&photon_data[base+11u]);
    s.color=photon_vec(base+12u);s.caster=atomicLoad(&photon_data[base+15u]);return s;
}
fn write_reservoir_sample(base:u32,s:LightSample) {
    photon_write(base,s.position);atomicStore(&photon_data[base+3u],s.depth);
    photon_write(base+4u,s.normal);atomicStore(&photon_data[base+7u],s.valid);
    photon_write(base+8u,s.flux);atomicStore(&photon_data[base+11u],s.light);
    photon_write(base+12u,s.color);atomicStore(&photon_data[base+15u],s.caster);
}
@compute @workgroup_size(64)
fn photon_trace(@builtin(global_invocation_id) id:vec3<u32>) {
    let world=id.y;if id.x>=caustics.features.y {return;}
    if (params.transport.x==0u && world==1u)||(params.transport.x==1u && world==0u) {return;}
    feedback_world=world;
    var rng=hash(id.x^hash(params.size.w+0x419b2e17u)^hash(params.transport.z)^hash(world+131u));
    var selected_seed=hash(rng^0x6a09e667u);
    var selected=trace_light_sample(world,selected_seed);photon_feedback(selected);
    var total=photon_target(selected);var candidates=1.0;var weight=1.0;
    let bank=caustics.features.z%2u;let stride=caustics.offsets.x*2u*20u;
    if (PHOTON_METHODS&4u)!=0u && caustics.features.z%8u!=0u {
        // One temporal and two neighboring pool reservoirs, all in the same
        // primary-sample domain. No image-space shift or Jacobian is assumed.
        for(var source=0u;source<3u;source++) {
            var index=id.x;if source>0u {index=hash(id.x^hash(source*877u+caustics.features.z))%caustics.features.y;}
            let base=caustics.learning.w+(1u-bank)*stride+(world*caustics.offsets.x+index)*20u;
            let m=photon_float(base+2u);let source_weight=photon_float(base+1u);
            if m<=0.0 || source_weight<=0.0 {continue;}
            let seed=atomicLoad(&photon_data[base]);let sample=read_reservoir_sample(base+4u);
            atomicAdd(&photon_data[caustics.learning.x-6u+world*4u],1u);
            let confidence=min(m,0.25);let w=confidence*source_weight*photon_target(sample);
            total+=w;candidates+=confidence;
            if random(&rng)<w/total {selected=sample;selected_seed=seed;}
        }
        weight=total/(candidates*photon_target(selected));
    }
    if (PHOTON_METHODS&4u)!=0u {
    let reservoir=caustics.learning.w+bank*stride+(world*caustics.offsets.x+id.x)*20u;
    atomicStore(&photon_data[reservoir],selected_seed);
    atomicStore(&photon_data[reservoir+1u],bitcast<u32>(weight));
    atomicStore(&photon_data[reservoir+2u],bitcast<u32>(candidates));
    write_reservoir_sample(reservoir+4u,selected);
    }
    if selected.valid==0u {return;}
    let record=caustics.offsets.z+(world*caustics.offsets.x+id.x)*16u;
    let flux=selected.flux*(weight/f32(caustics.features.y));
    photon_write(record,selected.position);photon_write(record+3u,selected.normal);photon_write(record+6u,flux);
    atomicStore(&photon_data[record+9u],selected.depth);
    let bucket=photon_bucket(photon_cell(selected.position),world);
    atomicAdd(&photon_data[bucket],1u);
    atomicStore(&photon_data[record+10u],bucket);atomicStore(&photon_data[record+11u],1u);
    atomicStore(&photon_data[record+12u],selected.light);atomicStore(&photon_data[record+13u],selected.caster);
    let visible=photon_splat(world,selected.position,selected.normal,selected.color,flux,selected.depth);
    if (PHOTON_METHODS&1u)!=0u && selected.light<32u {
        let cell=guide_cell(world,selected.light,selected.caster);
        photon_add(caustics.learning.y+cell*2u+1u,min(10000.0,max(0.0,visible)));
    }
}
