@group(2) @binding(0) var<storage,read_write> tensor_data:array<f32>;
fn tensor_offsets()->vec4<u32> {
    let count=params.size.x*params.size.y*select(1u,2u,params.transport.x==2u);
    let capacity=((params.size.x*params.size.y*2u+31u)/32u)*32u;
    let masks=8u+2u*TENSOR_H*TENSOR_H+5u*TENSOR_H+3u;
    let features=masks+capacity;
    return vec4<u32>(masks,features,features+capacity*TENSOR_H,count);
}
@compute @workgroup_size(64)
fn tensor_pack(@builtin(global_invocation_id) id:vec3<u32>) {
    let o=tensor_offsets();let parameters=2u*TENSOR_H*TENSOR_H+5u*TENSOR_H+3u;
    if id.x==0u {
        tensor_data[0]=bitcast<f32>(o.w);tensor_data[1]=bitcast<f32>(TENSOR_H);
        tensor_data[2]=bitcast<f32>(8u);tensor_data[3]=bitcast<f32>(o.x);
        tensor_data[4]=bitcast<f32>(o.y);tensor_data[5]=bitcast<f32>(o.z);
        tensor_data[6]=TENSOR_CLAMP;tensor_data[7]=0.0;
    }
    if id.x<parameters {tensor_data[8u+id.x]=tensor_weight(id.x);}
    if id.x>=o.w {return;}
    let s=neural_queries[id.x];let valid=tensor_valid(s);
    tensor_data[o.x+id.x]=select(0.0,1.0,valid);
    if !valid {return;}
    for(var f=0u;f<TENSOR_H;f++) {
        tensor_data[o.y+id.x*TENSOR_H+f]=tensor_feature(s,f);
    }
}
fn tensor_prediction(index:u32)->vec3<f32> {
    let o=tensor_offsets();let p=o.z+index*3u;
    return vec3<f32>(tensor_data[p],tensor_data[p+1u],tensor_data[p+2u]);
}
