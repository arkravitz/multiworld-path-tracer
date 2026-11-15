// Fixed-budget systematic resampling of pixels. Bounded priorities keep every
// pixel discoverable and bound rays per active invocation. All scans and queue
// compaction run on the GPU, with no host readback of scheduling decisions.
var<workgroup> adaptive_scan:array<f32,256>;
var<workgroup> adaptive_scan_u:array<u32,256>;
fn adaptive_prefix(lane:u32) {
    workgroupBarrier();
    for(var stride=1u;stride<256u;stride*=2u){
        var add=0.0;if lane>=stride {add=adaptive_scan[lane-stride];}
        workgroupBarrier();adaptive_scan[lane]+=add;workgroupBarrier();
    }
}
fn adaptive_prefix_u(lane:u32) {
    workgroupBarrier();
    for(var stride=1u;stride<256u;stride*=2u){
        var add=0u;if lane>=stride {add=adaptive_scan_u[lane-stride];}
        workgroupBarrier();adaptive_scan_u[lane]+=add;workgroupBarrier();
    }
}
fn adaptive_luma(v:vec3<f32>)->f32 {return dot(v,vec3<f32>(0.2126,0.7152,0.0722));}
fn adaptive_signal(mean:vec4<f32>, count:f32, cost:f32, prior:f32)->f32 {
    let standard=sqrt(max(0.0,mean.w)/max(1.0,count-1.0));
    var signal=standard;
    if params.reconstruction.y>1.5 && prior>0.0 {
        // Half-image denoiser disagreement measures the uncertainty remaining
        // AFTER reconstruction. Retain a raw-variance guard for missed events.
        signal=max(standard*0.1,sqrt(prior));
    }
    return signal/(max(0.03,adaptive_luma(max(vec3<f32>(0.0),mean.xyz)))*sqrt(max(1.0,cost/max(1.0,count))));
}
@compute @workgroup_size(64)
fn adaptive_reset(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=params.size.x*params.size.y {return;}
    pixels[id.x]=Pixel();
}
@compute @workgroup_size(256)
fn adaptive_scores(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>) {
    let n=params.size.x*params.size.y;var weight=0.0;
    if id.x<n {
        let a=adaptive_signal(pixels[id.x].moment_a,pixels[id.x].a.w,pixels[id.x].stats_a.w,pixels[id.x].guide_normal_a.w);
        var signal=a;
        if params.transport.x==2u {
            let b=adaptive_signal(pixels[id.x].moment_b,pixels[id.x].b.w,pixels[id.x].stats_b.w,pixels[id.x].guide_normal_b.w);
            signal=mix(b,a,params.controls.x);
        }
        weight=0.2+0.8*signal/(signal+0.25);
        if params.size.w<u32(params.reconstruction.z) {weight=1.0;}
    }
    adaptive_scan[lane]=weight;adaptive_prefix(lane);
    if id.x<n {pixels[id.x].allocation.y=adaptive_scan[lane];}
    if lane==255u {pixels[group.x].allocation.z=adaptive_scan[lane];}
}
@compute @workgroup_size(256)
fn adaptive_weight_blocks(@builtin(local_invocation_index) lane:u32) {
    let blocks=(params.size.x*params.size.y+255u)/256u;var carry=0.0;
    for(var base=0u;base<blocks;base+=256u){
        var value=0.0;if base+lane<blocks {value=pixels[base+lane].allocation.z;}
        adaptive_scan[lane]=value;adaptive_prefix(lane);
        if base+lane<blocks {pixels[base+lane].allocation.w=carry+adaptive_scan[lane]-value;}
        carry+=adaptive_scan[255u];workgroupBarrier();
    }
    if lane==0u {pixels[0].allocation.x=carry;}
}
@compute @workgroup_size(256)
fn adaptive_allocate(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>) {
    let n=params.size.x*params.size.y;var count=0u;
    if id.x<n {
        let scale=f32(n)/max(1e-20,pixels[0].allocation.x);
        let block=pixels[group.x].allocation.w;let right=(block+pixels[id.x].allocation.y)*scale;
        // Read the preceding endpoint even across a block boundary. Different
        // floating-point scan sums must not leave gaps in the fixed budget.
        var left=0.0;if id.x>0u {left=(pixels[(id.x-1u)/256u].allocation.w+pixels[id.x-1u].allocation.y)*scale;}
        var upper=right;if id.x+1u==n {upper=f32(n);}
        if params.size.w<u32(params.reconstruction.z) {upper=f32(id.x+1u);left=f32(id.x);}
        let jitter=f32(hash(params.transport.z^hash(params.size.w)^0x38f96be5u)>>8u)*(1.0/16777216.0);
        count=u32(floor(upper+jitter))-u32(floor(left+jitter));
        pixels[id.x].schedule.x=count;pixels[id.x].schedule.y=bitcast<u32>(max(1e-6,upper-left));
        // Add predictable control variates to EVERY pixel, even an inactive one.
        // Active rays later add inverse-rate-weighted residuals. This avoids the
        // optional-stopping bias of a self-normalized adaptive pixel average.
        if params.size.w>=u32(params.reconstruction.z) {
            pixels[id.x].a+=vec4<f32>(pixels[id.x].moment_a.xyz,0.0);
            pixels[id.x].stats_a+=vec4<f32>(pixels[id.x].stats_a.xyz/f32(params.size.w),0.0);
            pixels[id.x].normals_a+=vec4<f32>(pixels[id.x].normals_a.xyz/f32(params.size.w),0.0);
            if params.transport.x==2u {
                pixels[id.x].b+=vec4<f32>(pixels[id.x].moment_b.xyz,0.0);
                pixels[id.x].stats_b+=vec4<f32>(pixels[id.x].stats_b.xyz/f32(params.size.w),0.0);
                pixels[id.x].normals_b+=vec4<f32>(pixels[id.x].normals_b.xyz/f32(params.size.w),0.0);
            }
        }
        let a=caustic_visible(id.x,select(0u,1u,params.transport.x==1u));
        pixels[id.x].a+=vec4<f32>(a,0.0);pixels[id.x].light_a+=vec4<f32>(a,1.0);
        if params.transport.x==2u {let b=caustic_visible(id.x,1u);pixels[id.x].b+=vec4<f32>(b,0.0);pixels[id.x].light_b+=vec4<f32>(b,1.0);}
    }
    let chosen=select(0u,1u,count>0u);adaptive_scan_u[lane]=chosen;adaptive_prefix_u(lane);
    if id.x<n {pixels[id.x].schedule.z=adaptive_scan_u[lane]-chosen;}
    if lane==255u {pixels[group.x].schedule.w=adaptive_scan_u[lane];}
}
@compute @workgroup_size(256)
fn adaptive_active_blocks(@builtin(local_invocation_index) lane:u32) {
    let blocks=(params.size.x*params.size.y+255u)/256u;var carry=0u;
    for(var base=0u;base<blocks;base+=256u){
        var value=0u;if base+lane<blocks {value=pixels[base+lane].schedule.w;}
        adaptive_scan_u[lane]=value;adaptive_prefix_u(lane);
        if base+lane<blocks {pixels[base+lane].schedule.w=carry+adaptive_scan_u[lane]-value;}
        carry+=adaptive_scan_u[255u];workgroupBarrier();
    }
    if lane==0u {pixels[0].allocation.x=f32(min(carry,params.size.x*params.size.y));pixels[0].work.y=(min(carry,params.size.x*params.size.y)+63u)/64u;pixels[0].work.z=1u;pixels[0].work.w=1u;}
}
@compute @workgroup_size(64)
fn adaptive_compact(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=params.size.x*params.size.y {return;}
    if pixels[id.x].schedule.x==0u {return;}
    let slot=pixels[id.x].schedule.z+pixels[id.x/256u].schedule.w;
    pixels[slot].work.x=id.x;
}
@compute @workgroup_size(8,8)
fn adaptive_guides(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=params.size.x || id.y>=params.size.y {return;}
    let index=id.y*params.size.x+id.x;
    frame_store.frames[index]=guide_frame(id,linear_pixel(pixels[index],f32(max(1u,params.size.w+params.size.z))),1.0);
}
fn observe_camera(index:u32, result:Result, world:u32) {
    if world==0u {
        let previous=pixels[index].counts.x;let count=f32(previous);let mean=pixels[index].moment_a.xyz;let next=mean+(result.radiance-mean)/(count+1.0);
        let delta=adaptive_luma(result.radiance-mean);let residual=adaptive_luma(result.radiance-next);
        pixels[index].moment_a=vec4<f32>(next,max(0.0,pixels[index].moment_a.w+delta*residual));pixels[index].counts.x=previous+1u;pixels[index].a.w=f32(previous+1u);
        if (previous&1u)==0u {pixels[index].half_a=vec4<f32>(pixels[index].half_a.xyz+result.radiance,f32(previous/2u+1u));}
        pixels[index].stats_a.w+=result.stats.z;
        pixels[index].guide_normal_a+=vec4<f32>(result.guide_normal,0.0);pixels[index].guide_albedo_a+=vec4<f32>(result.guide_albedo,1.0);
    }else{
        let previous=pixels[index].counts.y;let count=f32(previous);let mean=pixels[index].moment_b.xyz;let next=mean+(result.radiance-mean)/(count+1.0);
        let delta=adaptive_luma(result.radiance-mean);let residual=adaptive_luma(result.radiance-next);
        pixels[index].moment_b=vec4<f32>(next,max(0.0,pixels[index].moment_b.w+delta*residual));pixels[index].counts.y=previous+1u;pixels[index].b.w=f32(previous+1u);
        if (previous&1u)==0u {pixels[index].half_b=vec4<f32>(pixels[index].half_b.xyz+result.radiance,f32(previous/2u+1u));}
        pixels[index].stats_b.w+=result.stats.z;
        pixels[index].guide_normal_b+=vec4<f32>(result.guide_normal,0.0);pixels[index].guide_albedo_b+=vec4<f32>(result.guide_albedo,1.0);
    }
}
