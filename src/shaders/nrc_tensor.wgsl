const TENSOR_H:u32=32u;
const TENSOR_CLAMP:f32=64.0;
fn tensor_weight(index:u32)->f32 {return neural_prediction_weight(index);}
fn tensor_feature(s:NeuralSample,index:u32)->f32 {return neural_feature(s,index);}
fn tensor_valid(s:NeuralSample)->bool {return s.label.w>0.0;}
@compute @workgroup_size(64)
fn tensor_apply(@builtin(global_invocation_id) id:vec3<u32>) {
    let slots=select(1u,2u,params.transport.x==2u);
    if id.x>=params.size.x*params.size.y*slots {return;}
    let s=neural_queries[id.x];if !tensor_valid(s) {return;}
    let index=id.x/slots;let color=s.label.xyz*tensor_prediction(id.x);
    neural_queries[id.x].label=vec4<f32>(color,1.0);
    let delta=adaptive_luma(color);let prefix=s.albedo.w;
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
