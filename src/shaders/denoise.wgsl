// Display-only, dual-world temporal reprojection with a small geometry-guided
// spatial denoise_params. No history or filtered value is ever written to `pixels`.
struct FilterParams {
    previous_eye:vec4<f32>, previous_forward:vec4<f32>, previous_right:vec4<f32>, previous_up:vec4<f32>,
    flags:vec4<u32>, // valid history, camera moved, unused, unused
    controls:vec4<f32>, // strength, motion history sample cap, unused, unused
}
@group(1) @binding(0) var<storage,read> history_previous:array<Frame>;
@group(1) @binding(1) var<storage,read_write> history_next:array<Frame>;
@group(1) @binding(2) var<uniform> denoise_params:FilterParams;
fn footprint(position:vec3<f32>) -> f32 {
    return max(0.003,length(position-params.eye.xyz)*length(params.up.xyz)*2.0/f32(params.size.y));
}
fn surface_compatible(p:vec4<f32>, n:vec4<f32>, a:vec4<f32>, q:vec4<f32>, m:vec4<f32>, b:vec4<f32>, temporal:bool) -> bool {
    if p.w!=q.w {return false;}
    if p.w==0.0 {return true;}
    if temporal && denoise_params.flags.y!=0u && (n.w>0.5 || m.w>0.5) {return false;}
    if dot(n.xyz,m.xyz)<0.95 {return false;}
    if length(a.xyz-b.xyz)>0.025 || abs(a.w-b.w)>0.05 {return false;}
    return distance(p.xyz,q.xyz)<=4.0*footprint(p.xyz);
}
fn compatible(a:Frame,b:Frame,temporal:bool) -> bool {
    if params.transport.x!=1u && !surface_compatible(a.position_a,a.normal_a,a.albedo_a,b.position_a,b.normal_a,b.albedo_a,temporal) {return false;}
    if params.transport.x!=0u && !surface_compatible(a.position_b,a.normal_b,a.albedo_b,b.position_b,b.normal_b,b.albedo_b,temporal) {return false;}
    return true;
}
fn previous_projection(position:vec3<f32>) -> vec3<f32> {
    let v=position-denoise_params.previous_eye.xyz;
    let depth=dot(v,denoise_params.previous_forward.xyz);
    if depth<=0.0001 {return vec3<f32>(-10000.0,-10000.0,-1.0);}
    let x=dot(v,normalize(denoise_params.previous_right.xyz))/(depth*length(denoise_params.previous_right.xyz));
    let y=dot(v,normalize(denoise_params.previous_up.xyz))/(depth*length(denoise_params.previous_up.xyz));
    return vec3<f32>((x*0.5+0.5)*f32(params.size.x)-0.5,(0.5-y*0.5)*f32(params.size.y)-0.5,depth);
}
fn projected_history(id:vec3<u32>,frame:Frame) -> vec3<f32> {
    var a=vec3<f32>(0.0);var b=a;var count=0u;
    if params.transport.x!=1u && frame.position_a.w>0.0 {a=previous_projection(frame.position_a.xyz);count++;}
    if params.transport.x!=0u && frame.position_b.w>0.0 {b=previous_projection(frame.position_b.xyz);count++;}
    if count==2u {
        // A hybrid pixel represents two potentially different depths. Reusing a
        // single history sample is safe only when BOTH projections agree.
        if a.z<=0.0 || b.z<=0.0 || distance(a.xy,b.xy)>1.5 {return vec3<f32>(-10000.0,-10000.0,-1.0);}
        return (a+b)*0.5;
    }
    if count==1u {if a.z>0.0 {return a;}return b;}
    let uv=(vec2<f32>(id.xy)+0.5)/vec2<f32>(params.size.xy);
    let direction=normalize(params.forward.xyz+(2.0*uv.x-1.0)*params.right.xyz+(1.0-2.0*uv.y)*params.up.xyz);
    return previous_projection(denoise_params.previous_eye.xyz+direction);
}
fn current_estimate(index:u32,from_raw:bool) -> vec3<f32> {
    if from_raw {return linear_pixel(pixels[index],f32(max(1u,params.size.w+params.size.z)));}
    return frames[index].sample.xyz;
}
@compute @workgroup_size(8,8)
fn denoise(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=params.size.x || id.y>=params.size.y {return;}
    let index=id.y*params.size.x+id.x;
    var current=frames[index];
    let total=params.size.w+params.size.z;
    if total==0u {
        current.sample=vec4<f32>(0.0);history_next[index]=current;
        display_estimate(id,pixels[index],vec3<f32>(0.0));return;
    }
    if denoise_params.controls.x<=0.0 {
        current.sample=vec4<f32>(linear_pixel(pixels[index],f32(total)),f32(total));
        history_next[index]=current;display_estimate(id,pixels[index],current.sample.xyz);return;
    }
    let projected=projected_history(id,current);
    let coordinate=vec2<i32>(floor(projected.xy+0.5));
    var valid=denoise_params.flags.x!=0u && projected.z>0.0 && all(coordinate>=vec2<i32>(0)) && all(coordinate<vec2<i32>(params.size.xy));
    var previous:Frame;
    if valid {
        previous=history_previous[u32(coordinate.y)*params.size.x+u32(coordinate.x)];
        valid=previous.sample.w>0.0 && compatible(current,previous,true);
    }
    if params.size.z==0u && valid && denoise_params.flags.y==0u {
        // Exposure/view-only updates must not accumulate the same samples twice.
        current.sample=previous.sample;history_next[index]=current;
        display_estimate(id,pixels[index],current.sample.xyz);return;
    }
    // No history (toggle/reset/headless final image): start with the complete raw
    // progressive estimate. Otherwise consume ONLY this dispatch's new samples.
    let from_raw=!valid || params.size.z==0u;
    let center=current_estimate(index,from_raw);
    var sum=vec3<f32>(0.0);var square=sum;var weight=0.0;
    for (var dy=-1;dy<=1;dy++) {for (var dx=-1;dx<=1;dx++) {
        let q=vec2<i32>(id.xy)+vec2<i32>(dx,dy);
        if any(q<vec2<i32>(0)) || any(q>=vec2<i32>(params.size.xy)) {continue;}
        let j=u32(q.y)*params.size.x+u32(q.x);
        if !compatible(current,frames[j],false) {continue;}
        let w=select(1.0,2.0,dx==0 && dy==0);
        let value=current_estimate(j,from_raw);
        sum+=w*value;square+=w*value*value;weight+=w;
    }}
    let mean=sum/max(weight,1.0);
    let variance=max(vec3<f32>(0.0),square/max(weight,1.0)-mean*mean);
    let fresh=mix(center,mean,denoise_params.controls.x);
    var count=select(f32(params.size.z),f32(total),from_raw);
    var estimate=fresh;
    if valid && !from_raw {
        let extent=3.0*sqrt(variance)+vec3<f32>(0.02);
        var old=previous.sample.xyz;
        if denoise_params.flags.y!=0u {old=clamp(old,mean-extent,mean+extent);}
        var old_count=min(previous.sample.w,1000000.0);
        if denoise_params.flags.y!=0u {old_count=min(old_count,denoise_params.controls.y);}
        if denoise_params.flags.y!=0u {old_count*=denoise_params.controls.x;}
        estimate=(fresh*count+old*old_count)/max(count+old_count,1.0);
        count+=old_count;
    }
    current.sample=vec4<f32>(max(estimate,vec3<f32>(0.0)),count);
    history_next[index]=current;
    display_estimate(id,pixels[index],current.sample.xyz);
}
