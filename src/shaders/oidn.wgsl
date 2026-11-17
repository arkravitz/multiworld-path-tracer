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
struct OidnSnapshot {size:vec4<u32>,controls:vec4<f32>}
@group(1) @binding(6) var<uniform> oidn_snapshot:OidnSnapshot;

fn oidn_native_variance(even:vec4<f32>,odd:vec4<f32>,clean_even:vec3<f32>,clean_odd:vec3<f32>)->f32 {
    let n=even.w+odd.w;
    if oidn_snapshot.size.w==0u || n<=1.0 {return -1.0;}
    let x=mix(even.xyz,clean_even,oidn_snapshot.controls.x);
    let y=mix(odd.xyz,clean_odd,oidn_snapshot.controls.x);
    return max(0.0,even.w*odd.w/n*dot((x-y)*(x-y),vec3<f32>(0.2126,0.7152,0.0722)));
}
// OIDN writes to the second half of shared Metal storage. Publication copies
// only a completed, current-generation snapshot into the displayed buffer.
@compute @workgroup_size(8,8)
fn oidn_finalize(@builtin(global_invocation_id) id:vec3<u32>) {
    let size=oidn_snapshot.size.xy;
    if any(id.xy>=size) {return;}
    let i=id.y*size.x+id.x;
    let raw=oidn_image[i];let clean=oidn_image[size.x*size.y+i];
    var p=OidnFeedback(vec4<f32>(clean.color_a.xyz,raw.color_a.w),raw.color_a,raw.color_b,raw.color_b);
    p.raw_a.w=oidn_native_variance(raw.half_a0,raw.half_a1,clean.half_a0.xyz,clean.half_a1.xyz);
    p.raw_b.w=-1.0;
    if oidn_snapshot.size.z!=0u {
        p.color_b=vec4<f32>(clean.color_b.xyz,raw.color_b.w);
        p.raw_b.w=oidn_native_variance(raw.half_b0,raw.half_b1,clean.half_b0.xyz,clean.half_b1.xyz);
    }
    oidn_feedback[i]=p;
}
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
@compute @workgroup_size(8,8)
fn oidn_present(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=params.size.x || id.y>=params.size.y {return;}
    let index=id.y*params.size.x+id.x;let p=pixels[index];let image=oidn_feedback[index];
    // Match an export of the completed snapshot, including its raw/filtered
    // strength blend. New raw paths must not reintroduce grain while OIDN works.
    // The renderer uses ordinary presentation until this generation has an image.
    var raw=image.raw_a.xyz;var filtered=image.color_a.xyz;
    if params.transport.x==2u {
        raw=mix(image.raw_b.xyz,raw,params.controls.x);
        filtered=mix(image.color_b.xyz,filtered,params.controls.x);
    }
    display_estimate(id,p,mix(raw,max(vec3<f32>(0.0),filtered),params.reconstruction.x));
}
