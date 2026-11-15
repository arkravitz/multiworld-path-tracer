// One SIMD group evaluates 32 queries, keeping layer tensors in threadgroup
// memory. Matrix fragments use Metal SIMD matrix instructions via Naga.
var<workgroup> nrc_ping:array<f32,1024>;
var<workgroup> nrc_pong:array<f32,1024>;
var<workgroup> nrc_mask:array<u32,32>;
var<workgroup> nrc_any:u32;
fn neural_resolve_matrix(lane:u32)->vec3<f32> {
    nrc_mask[lane]=select(0u,1u,neural_pending);
    workgroupBarrier();
    if lane==0u {
        var mask=0u;for(var i=0u;i<32u;i++) {mask+=nrc_mask[i];}nrc_any=mask;
    }
    let live=workgroupUniformLoad(&nrc_any);
    if live==0u {return vec3<f32>(0.0);}
    var rank=0u;for(var i=0u;i<lane;i++) {rank+=nrc_mask[i];}
    for(var i=0u;i<32u;i++) {
        if neural_pending {nrc_ping[rank*32u+i]=neural_feature(neural_pending_sample,i);}
        if lane>=live {nrc_ping[lane*32u+i]=0.0;}
        nrc_pong[lane*32u+i]=neural_prediction_weight(1024u+i);
    }
    workgroupBarrier();
    for(var row=0u;row<4u;row++) {
        var a0=coopLoad<coop_mat8x8<f32,A>>(&frame_store.neural.weights[row*8u],32u);
        if NRC_TEACHER {a0=coopLoad<coop_mat8x8<f32,A>>(&frame_store.neural.training_weights[row*8u],32u);}
        var a1=coopLoad<coop_mat8x8<f32,A>>(&frame_store.neural.weights[256u+row*8u],32u);
        if NRC_TEACHER {a1=coopLoad<coop_mat8x8<f32,A>>(&frame_store.neural.training_weights[256u+row*8u],32u);}
        var a2=coopLoad<coop_mat8x8<f32,A>>(&frame_store.neural.weights[512u+row*8u],32u);
        if NRC_TEACHER {a2=coopLoad<coop_mat8x8<f32,A>>(&frame_store.neural.training_weights[512u+row*8u],32u);}
        var a3=coopLoad<coop_mat8x8<f32,A>>(&frame_store.neural.weights[768u+row*8u],32u);
        if NRC_TEACHER {a3=coopLoad<coop_mat8x8<f32,A>>(&frame_store.neural.training_weights[768u+row*8u],32u);}
        for(var col=0u;col<(live+7u)/8u;col++) {
            let base=col*256u;
            let b0=coopLoad<coop_mat8x8<f32,B>>(&nrc_ping[base],32u);
            let b1=coopLoad<coop_mat8x8<f32,B>>(&nrc_ping[base+8u],32u);
            let b2=coopLoad<coop_mat8x8<f32,B>>(&nrc_ping[base+16u],32u);
            let b3=coopLoad<coop_mat8x8<f32,B>>(&nrc_ping[base+24u],32u);
            var c=coopLoad<coop_mat8x8<f32,C>>(&nrc_pong[base+row*8u],32u);
            c=coopMultiplyAdd(a0,b0,c);c=coopMultiplyAdd(a1,b1,c);
            c=coopMultiplyAdd(a2,b2,c);c=coopMultiplyAdd(a3,b3,c);
            coopStore(c,&nrc_pong[base+row*8u],32u);
        }
    }
    workgroupBarrier();
    for(var i=0u;i<32u;i++) {
        nrc_pong[lane*32u+i]=max(0.0,nrc_pong[lane*32u+i]);
        nrc_ping[lane*32u+i]=neural_prediction_weight(2080u+i);
    }
    workgroupBarrier();
    for(var row=0u;row<4u;row++) {
        var a0=coopLoad<coop_mat8x8<f32,A>>(&frame_store.neural.weights[1056u+row*8u],32u);
        if NRC_TEACHER {a0=coopLoad<coop_mat8x8<f32,A>>(&frame_store.neural.training_weights[1056u+row*8u],32u);}
        var a1=coopLoad<coop_mat8x8<f32,A>>(&frame_store.neural.weights[1312u+row*8u],32u);
        if NRC_TEACHER {a1=coopLoad<coop_mat8x8<f32,A>>(&frame_store.neural.training_weights[1312u+row*8u],32u);}
        var a2=coopLoad<coop_mat8x8<f32,A>>(&frame_store.neural.weights[1568u+row*8u],32u);
        if NRC_TEACHER {a2=coopLoad<coop_mat8x8<f32,A>>(&frame_store.neural.training_weights[1568u+row*8u],32u);}
        var a3=coopLoad<coop_mat8x8<f32,A>>(&frame_store.neural.weights[1824u+row*8u],32u);
        if NRC_TEACHER {a3=coopLoad<coop_mat8x8<f32,A>>(&frame_store.neural.training_weights[1824u+row*8u],32u);}
        for(var col=0u;col<(live+7u)/8u;col++) {
            let base=col*256u;
            let b0=coopLoad<coop_mat8x8<f32,B>>(&nrc_pong[base],32u);
            let b1=coopLoad<coop_mat8x8<f32,B>>(&nrc_pong[base+8u],32u);
            let b2=coopLoad<coop_mat8x8<f32,B>>(&nrc_pong[base+16u],32u);
            let b3=coopLoad<coop_mat8x8<f32,B>>(&nrc_pong[base+24u],32u);
            var c=coopLoad<coop_mat8x8<f32,C>>(&nrc_ping[base+row*8u],32u);
            c=coopMultiplyAdd(a0,b0,c);c=coopMultiplyAdd(a1,b1,c);
            c=coopMultiplyAdd(a2,b2,c);c=coopMultiplyAdd(a3,b3,c);
            coopStore(c,&nrc_ping[base+row*8u],32u);
        }
    }
    workgroupBarrier();
    var result=vec3<f32>(0.0);
    for(var channel=0u;channel<3u;channel++) {
        var sum=neural_prediction_weight(2208u+channel);
        for(var j=0u;j<32u;j++) {sum+=max(0.0,nrc_ping[rank*32u+j])*neural_prediction_weight(2112u+j*3u+channel);}
        result[channel]=clamp(sum,0.0,64.0);
    }
    return result;
}

// Coherent neural work follows traversal. Each invocation owns one RGB field;
// blend's two slots update disjoint a/b fields, with no shared component writes.
@compute @workgroup_size(32)
fn neural_resolve_queries(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let slots=select(1u,2u,ACTIVE_MODE==2u);
    let live=id.x<params.size.x*params.size.y*slots;
    neural_pending=false;
    if live {neural_pending_sample=neural_queries[id.x];neural_pending=neural_pending_sample.label.w>0.0;}
    let prediction=neural_resolve_matrix(lane);
    if !neural_pending {return;}
    let index=id.x/slots;let color=neural_pending_sample.label.xyz*prediction;
    neural_queries[id.x].label=vec4<f32>(color,1.0);
    let delta=adaptive_luma(color);let prefix=neural_pending_sample.albedo.w;
    if slots==2u && (id.x&1u)!=0u {
        let count=max(1.0,pixels[index].b.w);let old=pixels[index].moment_b;
        pixels[index].moment_b=vec4<f32>(old.xyz+color/count,max(0.0,old.w+2.0*delta*(prefix-adaptive_luma(old.xyz))+delta*delta*(1.0-1.0/count)));
        if (pixels[index].counts.y&1u)==1u {pixels[index].half_b+=vec4<f32>(color,0.0);}
        pixels[index].b+=vec4<f32>(color,0.0);
    }else{
        let count=max(1.0,pixels[index].a.w);let old=pixels[index].moment_a;
        pixels[index].moment_a=vec4<f32>(old.xyz+color/count,max(0.0,old.w+2.0*delta*(prefix-adaptive_luma(old.xyz))+delta*delta*(1.0-1.0/count)));
        if (pixels[index].counts.x&1u)==1u {pixels[index].half_a+=vec4<f32>(color,0.0);}
        pixels[index].a+=vec4<f32>(color,0.0);
    }
}

@compute @workgroup_size(8,8)
fn neural_finish_guides(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=params.size.x || id.y>=params.size.y {return;}
    let index=id.y*params.size.x+id.x;let slots=select(1u,2u,ACTIVE_MODE==2u);
    var color=vec3<f32>(0.0);
    for(var slot=0u;slot<slots;slot++) {
        let query=neural_queries[index*slots+slot];
        if query.label.w>0.0 {color+=query.label.xyz*select(1.0,select(params.controls.x,1.0-params.controls.x,slot==1u),slots==2u);}
    }
    frame_store.frames[index].sample+=vec4<f32>(color,0.0);
}

// The update pass batches its terminal predictions just like image queries.
// Prefix labels and terminal-to-prefix factors were written by teacher tracing.
@compute @workgroup_size(32)
fn neural_resolve_teachers(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    neural_pending_sample=frame_store.neural.teacher_queries[id.x];
    neural_pending=neural_pending_sample.label.w>0.0;
    let prediction=neural_resolve_matrix(lane);
    if !neural_pending {return;}
    for(var i=0u;i<4u;i++) {
        let index=id.x*4u+i;
        let sample=frame_store.neural.samples[index];
        if sample.label.w>0.0 && sample.albedo.w==1.0 {
            let label=max(vec3<f32>(0.0),sample.label.xyz+frame_store.neural.bootstrap_scales[index].xyz*prediction);
            frame_store.neural.samples[index].label=vec4<f32>(label,select(0.0,1.0,all(label<vec3<f32>(1e8))));
        }
    }
}
