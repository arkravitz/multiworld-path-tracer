// Inputs and completed feedback use separate buffers and layouts. A new
// readback can never overwrite the currently displayed reconstruction.
struct OidnInput {
    color_a:vec4<f32>, normal_a:vec4<f32>, albedo_a:vec4<f32>,
    color_b:vec4<f32>, normal_b:vec4<f32>, albedo_b:vec4<f32>,
    half_a0:vec4<f32>,half_a1:vec4<f32>,half_b0:vec4<f32>,half_b1:vec4<f32>,
}
struct OidnFeedback {color_a:vec4<f32>,raw_a:vec4<f32>,color_b:vec4<f32>,raw_b:vec4<f32>}
@group(1) @binding(4) var<storage,read_write> oidn_image:array<OidnInput>;
@group(1) @binding(5) var<storage,read_write> oidn_feedback:array<OidnFeedback>;
fn oidn_half(even:vec4<f32>,mean:vec4<f32>,count:f32,light:vec3<f32>,odd:bool)->vec4<f32> {
    var sum=even.xyz;var n=even.w;
    if odd {sum=mean.xyz*count-even.xyz;n=count-even.w;}
    // Both halves share the visible photon estimate: allocating extra camera
    // paths cannot reduce noise originating solely in independent light splats.
    return vec4<f32>(max(vec3<f32>(0.0),sum/max(1.0,n)+light),n);
}
@compute @workgroup_size(8,8)
fn oidn_pack(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=params.size.x || id.y>=params.size.y {return;}
    let index=id.y*params.size.x+id.x;let p=pixels[index];let n=f32(max(1u,params.size.w+params.size.z));
    oidn_image[index]=OidnInput(
        vec4<f32>(p.a.xyz/n,n),
        vec4<f32>(clamp(p.guide_normal_a.xyz/max(1.0,p.guide_albedo_a.w),vec3<f32>(-1.0),vec3<f32>(1.0)),0.0),
        vec4<f32>(clamp(p.guide_albedo_a.xyz/max(1.0,p.guide_albedo_a.w),vec3<f32>(0.0),vec3<f32>(1.0)),0.0),
        vec4<f32>(p.b.xyz/n,n),
        vec4<f32>(clamp(p.guide_normal_b.xyz/max(1.0,p.guide_albedo_b.w),vec3<f32>(-1.0),vec3<f32>(1.0)),0.0),
        vec4<f32>(clamp(p.guide_albedo_b.xyz/max(1.0,p.guide_albedo_b.w),vec3<f32>(0.0),vec3<f32>(1.0)),0.0),
        oidn_half(p.half_a,p.moment_a,p.a.w,p.light_a.xyz/n,false),oidn_half(p.half_a,p.moment_a,p.a.w,p.light_a.xyz/n,true),
        oidn_half(p.half_b,p.moment_b,p.b.w,p.light_b.xyz/n,false),oidn_half(p.half_b,p.moment_b,p.b.w,p.light_b.xyz/n,true));
}
@compute @workgroup_size(8,8)
fn oidn_publish(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=params.size.x || id.y>=params.size.y {return;}
    let index=id.y*params.size.x+id.x;var variance=vec2<f32>(0.0);var weights=vec2<f32>(0.0);
    // A small neighborhood stabilizes the error prior, not the rendered image.
    // It also directs discovery rays around isolated difficult pixels.
    for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
        let q=vec2<i32>(id.xy)+vec2<i32>(x,y);
        if any(q<vec2<i32>(0)) || any(q>=vec2<i32>(params.size.xy)) {continue;}
        let p=oidn_feedback[u32(q.y)*params.size.x+u32(q.x)];let weight=select(1.0,2.0,x==0 && y==0);
        if p.raw_a.w>=0.0 {variance.x+=p.raw_a.w*weight;weights.x+=weight;}
        if p.raw_b.w>=0.0 {variance.y+=p.raw_b.w*weight;weights.y+=weight;}
    }}
    pixels[index].guide_normal_a.w=select(0.0,max(1e-20,variance.x/max(1.0,weights.x)),weights.x>0.0);
    pixels[index].guide_normal_b.w=select(0.0,max(1e-20,variance.y/max(1.0,weights.y)),weights.y>0.0);
}
fn oidn_refined(filtered:vec4<f32>,snapshot:vec4<f32>,current:vec3<f32>,total:f32)->vec3<f32> {
    if filtered.w<=0.0 || filtered.w>total {return current/total;}
    // Preserve the weight of the filtered snapshot, adding subsequent raw
    // samples while a new reconstruction is in flight.
    return (filtered.xyz*filtered.w+current-snapshot.xyz*filtered.w)/total;
}
@compute @workgroup_size(8,8)
fn oidn_present(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=params.size.x || id.y>=params.size.y {return;}
    let index=id.y*params.size.x+id.x;let p=pixels[index];let image=oidn_feedback[index];
    let total=f32(max(1u,params.size.w+params.size.z));
    var color=oidn_refined(image.color_a,image.raw_a,p.a.xyz,total);
    if params.transport.x==2u {color=mix(oidn_refined(image.color_b,image.raw_b,p.b.xyz,total),color,params.controls.x);}
    display_estimate(id,p,mix(linear_pixel(p,total),max(vec3<f32>(0.0),color),params.reconstruction.x));
}
