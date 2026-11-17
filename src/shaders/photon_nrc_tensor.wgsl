const TENSOR_H:u32=64u;
const TENSOR_CLAMP:f32=1000.0;
fn tensor_weight(index:u32)->f32 {return sp.ema[index];}
fn tensor_feature(s:NeuralSample,index:u32)->f32 {return sp_feature(s,index,false);}
fn tensor_valid(s:NeuralSample)->bool {return s.label.w>0.0 && any(s.label.xyz!=vec3<f32>(0.0));}
@compute @workgroup_size(64)
fn tensor_apply(@builtin(global_invocation_id) id:vec3<u32>) {
    let slots=select(1u,2u,params.transport.x==2u);
    if id.x>=params.size.x*params.size.y*slots {return;}
    let s=neural_queries[id.x];if !tensor_valid(s) {return;}
    let color=s.label.xyz*tensor_prediction(id.x);
    neural_queries[id.x].label=vec4<f32>(color,s.label.w);
    sp_resolve_color(id.x,s,color);
}
