// Trainable 16-level, two-feature trilinear hash encoding; two 64-wide ReLU
// layers and RGB output. Frozen EMA inference, one fused minibatch update.
fn sp_grid_index(s:NeuralSample,level:u32,corner:u32)->u32 {
    let p=(s.position.xyz-sp_params.low.xyz)*sp_params.scale.xyz;
    let cell=vec3<i32>(floor(p*f32(16u<<level)))+vec3<i32>(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    return (level*SP_HASH_ENTRIES+(hash(bitcast<u32>(cell.x)^hash(bitcast<u32>(cell.y))^hash(bitcast<u32>(cell.z))^hash(u32(s.position.w)+19u))&(SP_HASH_ENTRIES-1u)))*2u;
}
fn sp_grid_weight(s:NeuralSample,level:u32,corner:u32)->f32 {
    let p=fract((s.position.xyz-sp_params.low.xyz)*sp_params.scale.xyz*f32(16u<<level));
    let w=select(vec3<f32>(1.0)-p,p,vec3<bool>((corner&1u)!=0u,((corner>>1u)&1u)!=0u,((corner>>2u)&1u)!=0u));return w.x*w.y*w.z;
}
fn sp_sh(n:vec3<f32>,i:u32)->f32 {
    switch i {
        case 0u:{return 0.2820948;}case 1u:{return -0.4886025*n.y;}case 2u:{return 0.4886025*n.z;}case 3u:{return -0.4886025*n.x;}
        case 4u:{return 1.0925484*n.x*n.y;}case 5u:{return -1.0925484*n.y*n.z;}case 6u:{return 0.3153916*(3.0*n.z*n.z-1.0);}
        case 7u:{return -1.0925484*n.x*n.z;}default:{return 0.5462742*(n.x*n.x-n.y*n.y);}
    }
}
fn sp_feature(s:NeuralSample,i:u32,training:bool)->f32 {
    if i<32u {
        let level=i/2u;let channel=i%2u;var value=0.0;
        for(var corner=0u;corner<8u;corner++){
            let index=sp_grid_index(s,level,corner)+channel;
            value+=sp_grid_weight(s,level,corner)*select(sp.hash_ema[index],sp.hash_weights[index],training);
        }return value;
    }
    if i<36u {return select(0.0,1.0,i==35u);} // all cached vertices are diffuse
    if i<45u {return sp_sh(s.normal.xyz,i-36u);} // diffuse outgoing radiance is direction independent
    if i<54u {return sp_sh(s.normal.xyz,i-45u);}
    if i<57u {return s.albedo[i-54u];}if i<60u {return 0.0;}
    if i==60u {return s.position.w;}if i==61u {return s.normal.w/64.0;}return select(0.0,1.0,i==63u);
}
fn sp_predict(s:NeuralSample)->vec3<f32>{
    var x:array<f32,64>;var a:array<f32,64>;var b:array<f32,64>;
    for(var i=0u;i<64u;i++){x[i]=sp_feature(s,i,false);}
    for(var i=0u;i<64u;i++){var v=sp.ema[4096u+i];for(var j=0u;j<64u;j++){v+=x[j]*sp.ema[j*64u+i];}a[i]=max(0.0,v);}
    for(var i=0u;i<64u;i++){var v=sp.ema[8256u+i];for(var j=0u;j<64u;j++){v+=a[j]*sp.ema[4160u+j*64u+i];}b[i]=max(0.0,v);}
    var out=vec3<f32>(0.0);for(var i=0u;i<3u;i++){var v=sp.ema[8512u+i];for(var j=0u;j<64u;j++){v+=b[j]*sp.ema[8320u+j*3u+i];}out[i]=clamp(v,0.0,1000.0);}return out;
}
@compute @workgroup_size(64)
fn sp_resolve(@builtin(global_invocation_id) id:vec3<u32>){
    let index=id.x+id.y*65535u*64u;
    let slots=select(1u,2u,params.transport.x==2u);if index>=params.size.x*params.size.y*slots {return;}
    let s=neural_queries[index];var color=vec3<f32>(0.0);
    if s.label.w>0.0 && any(s.label.xyz!=vec3<f32>(0.0)){color=s.label.xyz*sp_predict(s);}
    neural_queries[index].label=vec4<f32>(color,s.label.w);
    if all(color==vec3<f32>(0.0)){return;}
    sp_resolve_color(index,s,color);
}
fn sp_resolve_color(index:u32,s:NeuralSample,color:vec3<f32>) {
    let slots=select(1u,2u,params.transport.x==2u);
    let i=index/slots;let delta=adaptive_luma(color);let prefix=s.albedo.w;
    if slots==2u&&(index&1u)!=0u {
        let count=max(1.0,pixels[i].b.w);let old=pixels[i].moment_b;
        pixels[i].moment_b=vec4<f32>(old.xyz+color/count,max(0.0,old.w+2.0*delta*(prefix-adaptive_luma(old.xyz))+delta*delta*(1.0-1.0/count)));
        if (pixels[i].counts.y&1u)==1u {pixels[i].half_b+=vec4<f32>(color,0.0);}pixels[i].b+=vec4<f32>(color,0.0);
    }else{
        let count=max(1.0,pixels[i].a.w);let old=pixels[i].moment_a;
        pixels[i].moment_a=vec4<f32>(old.xyz+color/count,max(0.0,old.w+2.0*delta*(prefix-adaptive_luma(old.xyz))+delta*delta*(1.0-1.0/count)));
        if (pixels[i].counts.x&1u)==1u {pixels[i].half_a+=vec4<f32>(color,0.0);}pixels[i].a+=vec4<f32>(color,0.0);
    }
}
@compute @workgroup_size(64)
fn sp_finish_guides(@builtin(global_invocation_id) id:vec3<u32>){
    if id.x>=params.size.x*params.size.y{return;}let slots=select(1u,2u,params.transport.x==2u);var color=vec3<f32>(0.0);
    for(var slot=0u;slot<slots;slot++){let q=neural_queries[id.x*slots+slot];color+=q.label.xyz*select(1.0,select(params.controls.x,1.0-params.controls.x,slot==1u),slots==2u);}
    frame_store.frames[id.x].sample+=vec4<f32>(color,0.0);
}
var<workgroup> sp_x:array<f32,64>;var<workgroup> sp_a:array<f32,64>;var<workgroup> sp_b:array<f32,64>;
var<workgroup> sp_da:array<f32,64>;var<workgroup> sp_db:array<f32,64>;var<workgroup> sp_out:array<f32,3>;
fn sp_hash_add(index:u32,value:f32){
    if value==0.0{return;}var old=atomicLoad(&sp.hash_grad[index]);
    loop{let result=atomicCompareExchangeWeak(&sp.hash_grad[index],old,bitcast<u32>(bitcast<f32>(old)+value));if result.exchanged{break;}old=result.old_value;}
}
@compute @workgroup_size(64)
fn sp_train(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){
    let s=sp.training[group.x];sp_x[lane]=sp_feature(s,lane,true);workgroupBarrier();
    var a=sp.weights[4096u+lane];for(var j=0u;j<64u;j++){a+=sp_x[j]*sp.weights[j*64u+lane];}sp_a[lane]=max(0.0,a);workgroupBarrier();
    var b=sp.weights[8256u+lane];for(var j=0u;j<64u;j++){b+=sp_a[j]*sp.weights[4160u+j*64u+lane];}sp_b[lane]=max(0.0,b);workgroupBarrier();
    if lane<3u {var out=sp.weights[8512u+lane];for(var j=0u;j<64u;j++){out+=sp_b[j]*sp.weights[8320u+j*3u+lane];}sp_out[lane]=out;}workgroupBarrier();
    let luminance=dot(max(vec3<f32>(0.0),vec3<f32>(sp_out[0],sp_out[1],sp_out[2])),vec3<f32>(0.2126,0.7152,0.0722));
    if lane<3u {sp_out[lane]=(sp_out[lane]-s.label[lane])/(0.01+luminance*luminance)*s.label.w;}workgroupBarrier();
    var db=0.0;for(var j=0u;j<3u;j++){db+=sp.weights[8320u+lane*3u+j]*sp_out[j];}sp_db[lane]=select(0.0,db,sp_b[lane]>0.0);workgroupBarrier();
    var da=0.0;for(var j=0u;j<64u;j++){da+=sp.weights[4160u+lane*64u+j]*sp_db[j];}sp_da[lane]=select(0.0,da,sp_a[lane]>0.0);workgroupBarrier();
    let base=group.x*8515u;
    for(var j=0u;j<64u;j++){
        sp.gradients[base+j*64u+lane]=sp_da[lane]*sp_x[j];
        sp.gradients[base+4160u+j*64u+lane]=sp_db[lane]*sp_a[j];
        if lane<3u {sp.gradients[base+8320u+j*3u+lane]=sp_out[lane]*sp_b[j];}
    }
    sp.gradients[base+4096u+lane]=sp_da[lane];sp.gradients[base+8256u+lane]=sp_db[lane];
    if lane<3u {sp.gradients[base+8512u+lane]=sp_out[lane];}
    if lane<32u {
        var gradient=0.0;for(var j=0u;j<64u;j++){gradient+=sp.weights[lane*64u+j]*sp_da[j];}
        let level=lane/2u;for(var corner=0u;corner<8u;corner++){sp_hash_add(sp_grid_index(s,level,corner)+lane%2u,gradient*sp_grid_weight(s,level,corner));}
    }
}
@compute @workgroup_size(64)
fn sp_apply(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x+id.y*65536u;let count=f32(atomicLoad(&sp.state.valid));if count<=0.0{return;}
    let step=f32(sp.state.step+1u);let correction=vec2<f32>(1.0-pow(0.9,step),1.0-pow(0.99,step));
    if i<8515u {
        var g=0.0;for(var j=0u;j<128u;j++){g+=sp.gradients[j*8515u+i];}g=clamp(g/count+sp.weights[i]*1e-6,-10000.0,10000.0);
        let m=0.9*sp.m[i]+0.1*g;let v=0.99*sp.v[i]+0.01*g*g;sp.m[i]=m;sp.v[i]=v;
        let w=sp.weights[i]-0.01*(m/correction.x)/(sqrt(v/correction.y)+1e-15);sp.weights[i]=w;sp.ema[i]=mix(w,sp.ema[i],0.95);
    }
    if i<SP_HASH_PARAMETERS {
        let g=clamp(bitcast<f32>(atomicLoad(&sp.hash_grad[i]))/count,-10000.0,10000.0);
        let m=0.9*sp.hash_m[i]+0.1*g;let v=0.99*sp.hash_v[i]+0.01*g*g;sp.hash_m[i]=m;sp.hash_v[i]=v;
        let w=sp.hash_weights[i]*(1.0-0.01*1e-6)-0.01*(m/correction.x)/(sqrt(v/correction.y)+1e-15);sp.hash_weights[i]=w;sp.hash_ema[i]=mix(w,sp.hash_ema[i],0.95);atomicStore(&sp.hash_grad[i],0u);
    }
}
@compute @workgroup_size(1)
fn sp_finish(){sp.state.emitted+=sp_params.config.x;sp.state.round++;if atomicLoad(&sp.state.valid)>0u {sp.state.step++;}}
