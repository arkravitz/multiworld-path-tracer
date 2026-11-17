// Sixteen encoded queries per SIMD group. Both hidden layers use native Metal
// SIMD matrix operations; 8 KiB of shared activation storage, no global tensors.
var<workgroup> sp_ping:array<f32,1024>;
var<workgroup> sp_pong:array<f32,1024>;
var<workgroup> sp_mask:array<u32,32>;
var<workgroup> sp_any:u32;
@compute @workgroup_size(32)
fn sp_resolve_matrix(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){
    let slots=select(1u,2u,params.transport.x==2u);let index=(group.x+group.y*65535u)*16u+lane;
    var s:NeuralSample;var valid=false;
    if lane<16u&&index<params.size.x*params.size.y*slots {s=neural_queries[index];valid=s.label.w>0.0&&any(s.label.xyz!=vec3<f32>(0.0));}
    sp_mask[lane]=select(0u,1u,valid);workgroupBarrier();
    if lane==0u {var n=0u;for(var i=0u;i<32u;i++){n+=sp_mask[i];}sp_any=n;}
    let live=workgroupUniformLoad(&sp_any);if live==0u {return;}
    if lane<16u {for(var f=0u;f<64u;f++){var value=0.0;if valid {value=sp_feature(s,f,false);}sp_ping[lane*64u+f]=value;}}
    for(var i=lane;i<1024u;i+=32u){sp_pong[i]=sp.ema[4096u+i%64u];}workgroupBarrier();
    for(var row=0u;row<8u;row++){for(var col=0u;col<2u;col++){
        let base=col*512u;var c=coopLoad<coop_mat8x8<f32,C>>(&sp_pong[base+row*8u],64u);
        for(var k=0u;k<8u;k++){
            let a=coopLoad<coop_mat8x8<f32,A>>(&sp.ema[k*512u+row*8u],64u);
            let b=coopLoad<coop_mat8x8<f32,B>>(&sp_ping[base+k*8u],64u);c=coopMultiplyAdd(a,b,c);
        }coopStore(c,&sp_pong[base+row*8u],64u);
    }}workgroupBarrier();
    for(var i=lane;i<1024u;i+=32u){sp_pong[i]=max(0.0,sp_pong[i]);sp_ping[i]=sp.ema[8256u+i%64u];}workgroupBarrier();
    for(var row=0u;row<8u;row++){for(var col=0u;col<2u;col++){
        let base=col*512u;var c=coopLoad<coop_mat8x8<f32,C>>(&sp_ping[base+row*8u],64u);
        for(var k=0u;k<8u;k++){
            let a=coopLoad<coop_mat8x8<f32,A>>(&sp.ema[4160u+k*512u+row*8u],64u);
            let b=coopLoad<coop_mat8x8<f32,B>>(&sp_pong[base+k*8u],64u);c=coopMultiplyAdd(a,b,c);
        }coopStore(c,&sp_ping[base+row*8u],64u);
    }}workgroupBarrier();
    if !valid {return;}
    var prediction=vec3<f32>(0.0);
    for(var c=0u;c<3u;c++){
        var v=sp.ema[8512u+c];for(var j=0u;j<64u;j++){v+=max(0.0,sp_ping[lane*64u+j])*sp.ema[8320u+j*3u+c];}prediction[c]=clamp(v,0.0,1000.0);
    }
    let color=s.label.xyz*prediction;neural_queries[index].label=vec4<f32>(color,s.label.w);sp_resolve_color(index,s,color);
}
