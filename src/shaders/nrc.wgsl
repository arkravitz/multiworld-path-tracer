// Diffuse online NRC. Apple inference fuses SIMD matrix layers on chip; the
// portable fallback is scalar. Training fuses forward/backward per example, then
// reduces disjoint gradients in an ordered Adam/EMA dispatch.
// wgpu translates these compute entry points to Metal on Apple GPUs.
override NRC_ENABLED: bool = false;
override NRC_COOPERATIVE:bool=false;
override NRC_TEACHER:bool=false;
override NRC_QUEUED:bool=false;
override NRC_BOOTSTRAP:bool=true;
override NRC_TEACHER_QUEUED:bool=false;
override TRACE_GROUP_Y:u32=8u;
const NN_WIDTH: u32 = 32u;
const NN_PARAMETERS: u32 = 2211u;
const NN_RECORDS: u32 = 1024u;
const NN_BATCH:u32=128u;
struct NeuralSample {
    position: vec4<f32>, // offset origin, current world
    normal: vec4<f32>, // normal, remaining depth
    albedo: vec4<f32>, // xyz reflectance; teacher w: 1 bootstrapped, 2 uncached anchor
    label: vec4<f32>, // incident scattered RGB (albedo demodulated), validity
}
struct NeuralCache {
    state: vec4<u32>, // valid examples, Adam steps, minibatch serial, reserved
    weights: array<f32,2211>,
    first_moment: array<f32,2211>,
    second_moment: array<f32,2211>,
    samples: array<NeuralSample,1024>,
    training_weights:array<f32,2211>,
    gradients:array<f32,283008>,
    valid_samples:array<u32,128>,
    coefficients:vec4<f32>,
    teacher_queries:array<NeuralSample,256>,
    bootstrap_scales:array<vec4<f32>,1024>,
}
// Sharing the guide allocation avoids consuming an extra storage binding: the
// optional denoiser already reaches the eight-buffer baseline WebGPU limit.
struct FrameStore { neural: NeuralCache, frames: array<Frame> }
var<private> neural_teacher: bool;
var<private> neural_uncached_anchor:bool;
var<private> neural_pending:bool;
var<private> neural_pending_sample:NeuralSample;
var<private> neural_pending_scale:vec3<f32>;
var<private> neural_record_base: u32;

fn neural_ready()->bool {return frame_store.neural.state.x>=512u && frame_store.neural.state.y>=32u;}

// Triangle-wave positional encoding follows the paper's cheap periodic primitive.
fn neural_wave(phase:f32,shift:f32)->f32 {return 4.0*abs(fract(phase*0.15915494309+shift-0.25)-0.5)-1.0;}
fn neural_features(s: NeuralSample) -> array<f32,32> {
    var x: array<f32,32>;
    for (var a=0u;a<3u;a++) {
        x[a]=clamp(s.position[a]/8.0,-2.0,2.0);
        x[3u+a]=s.normal[a]; x[6u+a]=s.albedo[a];
        for (var f=0u;f<3u;f++) {
            let phase=s.position[a]*(0.5*f32(1u<<f));
            x[12u+f*6u+a*2u]=neural_wave(phase,0.0);
            x[13u+f*6u+a*2u]=neural_wave(phase,0.25);
        }
    }
    x[9]=s.position.w; x[10]=s.normal.w/64.0;
    x[11]=f32(params.transport.x)/5.0;
    x[30]=select(probability(s.position.xyz),0.5,params.transport.x<=3u); x[31]=params.controls.y;
    return x;
}
// Training targets use a snapshot of raw optimizer weights, not the lagging
// rendering EMA. Both snapshots remain immutable until teacher tracing ends.
fn neural_prediction_weight(index:u32)->f32 {
    return select(frame_store.neural.weights[index],frame_store.neural.training_weights[index],NRC_TEACHER);
}
fn neural_predict(s: NeuralSample) -> vec3<f32> {
    let x=neural_features(s);var a:array<f32,32>;var b:array<f32,32>;
    for (var i=0u;i<32u;i++) {
        var sum=neural_prediction_weight(1024u+i);
        for (var j=0u;j<32u;j++) {sum+=x[j]*neural_prediction_weight(j*32u+i);}
        a[i]=max(0.0,sum);
    }
    for (var i=0u;i<32u;i++) {
        var sum=neural_prediction_weight(2080u+i);
        for (var j=0u;j<32u;j++) {sum+=a[j]*neural_prediction_weight(1056u+j*32u+i);}
        b[i]=max(0.0,sum);
    }
    var result=vec3<f32>(0.0);
    for (var i=0u;i<3u;i++) {
        var sum=neural_prediction_weight(2208u+i);
        for (var j=0u;j<32u;j++) {sum+=b[j]*neural_prediction_weight(2112u+j*3u+i);}
        result[i]=clamp(sum,0.0,64.0);
    }
    return result;
}
@compute @workgroup_size(64)
fn neural_initialize(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x<NN_PARAMETERS {
        let bias=(id.x>=1024u && id.x<1056u) || (id.x>=2080u && id.x<2112u) || id.x>=2208u;
        let random_weight=(f32(hash(id.x^0x47ad92b1u)>>8u)/16777216.0-0.5)*0.35;
        frame_store.neural.weights[id.x]=select(random_weight,0.0,bias);
        frame_store.neural.training_weights[id.x]=select(random_weight,0.0,bias);
        frame_store.neural.first_moment[id.x]=0.0;
        frame_store.neural.second_moment[id.x]=0.0;
    }
    if id.x==0u {frame_store.neural.state=vec4<u32>(0u);}
}
@compute @workgroup_size(64)
fn neural_teach(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=256u {return;}
    neural_teacher=true; neural_record_base=id.x*4u; sequence_active=false;
    // Exactly 16 of 256 paths retain the ordinary finite-depth/roulette target.
    // Rotate membership so the anchor role is not tied to a teacher lane.
    neural_uncached_anchor=((id.x+params.neural.z)&15u)==0u;
    if NRC_TEACHER_QUEUED {frame_store.neural.teacher_queries[id.x].label=vec4<f32>(0.0); }
    for (var i=0u;i<4u;i++) {frame_store.neural.samples[neural_record_base+i].label=vec4<f32>(0.0);}
    let seed=hash(id.x^hash(params.neural.z)^hash(params.transport.z)^0xe927a3d5u);
    var rng=hash(seed^0x27c4d193u);
    let uv=vec2<f32>(random(&rng),random(&rng));
    let direction=normalize(params.forward.xyz+(2.0*uv.x-1.0)*params.right.xyz+(1.0-2.0*uv.y)*params.up.xyz);
    var mode=ACTIVE_MODE;
    if mode==2u {mode=id.x&1u;}
    let unused=trace(params.eye.xyz,direction,seed,mode);
}
var<workgroup> nn_weights:array<f32,2211>;
var<workgroup> nn_x:array<f32,32>;
var<workgroup> nn_a:array<f32,32>;
var<workgroup> nn_b:array<f32,32>;
var<workgroup> nn_da:array<f32,32>;
var<workgroup> nn_db:array<f32,32>;
var<workgroup> nn_dout:array<f32,3>;
var<workgroup> nn_step:u32;

fn neural_adam(index:u32, gradient:f32, correction:vec2<f32>) {
    let g=clamp(gradient,-10000.0,10000.0);
    let m=0.9*frame_store.neural.first_moment[index]+0.1*g;
    let v=0.999*frame_store.neural.second_moment[index]+0.001*g*g;
    frame_store.neural.first_moment[index]=m;frame_store.neural.second_moment[index]=v;
    nn_weights[index]-=0.001*(m/correction.x)/(sqrt(v/correction.y)+1e-8);
}
@compute @workgroup_size(32)
fn neural_train_serial_reference(@builtin(local_invocation_index) lane:u32) {
    // Column-major matrices coalesce neuron-lane loads and updates. Weights and
    // activations fit below 10 KiB of workgroup storage.
    // Exactly ONE workgroup may dispatch this entry. All lanes execute every
    // barrier, including invalid records. No cross-workgroup weight races.
    for (var i=lane;i<NN_PARAMETERS;i+=32u) {nn_weights[i]=frame_store.neural.weights[i];}
    if lane==0u {nn_step=frame_store.neural.state.x;}
    workgroupBarrier();
    for (var example=0u;example<16u;example++) {
        let record=hash(params.neural.z*16u+example+0x4ef37a29u)%NN_RECORDS;
        let sample=frame_store.neural.samples[record];let valid=sample.label.w;
        let features=neural_features(sample);nn_x[lane]=features[lane];
        workgroupBarrier();
        var first=nn_weights[1024u+lane];
        for (var j=0u;j<32u;j++) {first+=nn_x[j]*nn_weights[j*32u+lane];}
        nn_a[lane]=max(0.0,first);
        workgroupBarrier();
        var second=nn_weights[2080u+lane];
        for (var j=0u;j<32u;j++) {second+=nn_a[j]*nn_weights[1056u+j*32u+lane];}
        nn_b[lane]=max(0.0,second);
        workgroupBarrier();
        if lane<3u {
            var output=nn_weights[2208u+lane];
            for (var j=0u;j<32u;j++) {output+=nn_b[j]*nn_weights[2112u+j*3u+lane];}
            // Relative radiance L2: detach the prediction-dependent denominator.
            // RGB-space labels avoid the Jensen bias of log-target regression.
            // Full uncached continuations supply labels, never the network itself.
            let denominator=0.01+max(output,0.0)*max(output,0.0);
            nn_dout[lane]=(output-clamp(sample.label[lane],0.0,64.0))/denominator*valid;
        }
        workgroupBarrier();
        var db=0.0;
        for (var j=0u;j<3u;j++) {db+=nn_weights[2112u+lane*3u+j]*nn_dout[j];}
        nn_db[lane]=select(0.0,db,nn_b[lane]>0.0);
        workgroupBarrier();
        var da=0.0;
        for (var j=0u;j<32u;j++) {da+=nn_weights[1056u+lane*32u+j]*nn_db[j];}
        nn_da[lane]=select(0.0,da,nn_a[lane]>0.0);
        if lane==0u && valid>0.0 {nn_step++;}
        workgroupBarrier();
        // Every old weight has been read before any lane updates it.
        let correction=vec2<f32>(1.0-pow(0.9,f32(max(nn_step,1u))),1.0-pow(0.999,f32(max(nn_step,1u))));
        if valid>0.0 {
            for (var j=0u;j<32u;j++) {
                neural_adam(j*32u+lane,nn_da[lane]*nn_x[j],correction);
                neural_adam(1056u+j*32u+lane,nn_db[lane]*nn_a[j],correction);
                if lane<3u {neural_adam(2112u+j*3u+lane,nn_dout[lane]*nn_b[j],correction);}
            }
            neural_adam(1024u+lane,nn_da[lane],correction);
            neural_adam(2080u+lane,nn_db[lane],correction);
            if lane<3u {neural_adam(2208u+lane,nn_dout[lane],correction);}
        }
        // Workgroup and storage barriers order both on-chip activations and
        // global weights before the next example, within this sole workgroup.
        storageBarrier();workgroupBarrier();
    }
    for (var i=lane;i<NN_PARAMETERS;i+=32u) {frame_store.neural.weights[i]=nn_weights[i];}
    if lane==0u {frame_store.neural.state.x=nn_step;}
}

fn neural_feature(s:NeuralSample,index:u32)->f32 {
    if index<3u {return clamp(s.position[index]/8.0,-2.0,2.0);}
    if index<6u {return s.normal[index-3u];}
    if index<9u {return s.albedo[index-6u];}
    if index==9u {return s.position.w;}
    if index==10u {return s.normal.w/64.0;}
    if index==11u {return f32(params.transport.x)/5.0;}
    if index<30u {
        let feature=index-12u;let axis=(feature%6u)/2u;
        let phase=s.position[axis]*(0.5*f32(1u<<(feature/6u)));
        if (feature&1u)==0u {return neural_wave(phase,0.0);}return neural_wave(phase,0.25);
    }
    if index==30u {if params.transport.x<=3u {return 0.5;}return probability(s.position.xyz);}
    return params.controls.y;
}
// Fused forward/backward per example. Independent groups read frozen training
// weights and write disjoint partial gradients. Reduction/Adam follows later.
@compute @workgroup_size(32)
fn neural_train_fused(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>) {
    let record=(hash(params.neural.z^0x4ef37a29u)+(group.x+(frame_store.neural.state.z%4u)*NN_BATCH)*991u)%NN_RECORDS;
    let sample=frame_store.neural.samples[record];let valid=sample.label.w;
    for(var i=lane;i<NN_PARAMETERS;i+=32u) {nn_weights[i]=frame_store.neural.training_weights[i];}
    nn_x[lane]=neural_feature(sample,lane);
    if lane==0u {
        frame_store.neural.valid_samples[group.x]=select(0u,1u,valid>0.0);
        if group.x==0u {
            let step=f32(frame_store.neural.state.y+1u);
            frame_store.neural.coefficients=vec4<f32>(1.0-pow(0.9,step),1.0-pow(0.999,step),1.0-pow(0.99,step),1.0-pow(0.99,step-1.0));
        }
    }
    workgroupBarrier();
    var first=nn_weights[1024u+lane];
    for(var j=0u;j<32u;j++) {first+=nn_x[j]*nn_weights[j*32u+lane];}
    nn_a[lane]=max(0.0,first);workgroupBarrier();
    var second=nn_weights[2080u+lane];
    for(var j=0u;j<32u;j++) {second+=nn_a[j]*nn_weights[1056u+j*32u+lane];}
    nn_b[lane]=max(0.0,second);workgroupBarrier();
    if lane<3u {
        var output=nn_weights[2208u+lane];
        for(var j=0u;j<32u;j++) {output+=nn_b[j]*nn_weights[2112u+j*3u+lane];}
        nn_dout[lane]=(output-clamp(sample.label[lane],0.0,64.0))/(0.01+max(output,0.0)*max(output,0.0))*valid;
    }
    workgroupBarrier();var db=0.0;
    for(var j=0u;j<3u;j++) {db+=nn_weights[2112u+lane*3u+j]*nn_dout[j];}
    nn_db[lane]=select(0.0,db,nn_b[lane]>0.0);workgroupBarrier();var da=0.0;
    for(var j=0u;j<32u;j++) {da+=nn_weights[1056u+lane*32u+j]*nn_db[j];}
    nn_da[lane]=select(0.0,da,nn_a[lane]>0.0);
    let base=group.x*NN_PARAMETERS;
    for(var j=0u;j<32u;j++) {
        frame_store.neural.gradients[base+j*32u+lane]=nn_da[lane]*nn_x[j];
        frame_store.neural.gradients[base+1056u+j*32u+lane]=nn_db[lane]*nn_a[j];
        if lane<3u {frame_store.neural.gradients[base+2112u+j*3u+lane]=nn_dout[lane]*nn_b[j];}
    }
    frame_store.neural.gradients[base+1024u+lane]=nn_da[lane];
    frame_store.neural.gradients[base+2080u+lane]=nn_db[lane];
    if lane<3u {frame_store.neural.gradients[base+2208u+lane]=nn_dout[lane];}
}
@compute @workgroup_size(64)
fn neural_apply_gradients(@builtin(global_invocation_id) id:vec3<u32>) {
    let index=id.x;if index>=NN_PARAMETERS {return;}
    var gradient=0.0;var count=0u;
    for(var i=0u;i<NN_BATCH;i++) {gradient+=frame_store.neural.gradients[i*NN_PARAMETERS+index];count+=frame_store.neural.valid_samples[i];}
    if count==0u {return;}
    let g=clamp(gradient/f32(count),-10000.0,10000.0);
    let m=0.9*frame_store.neural.first_moment[index]+0.1*g;
    let v=0.999*frame_store.neural.second_moment[index]+0.001*g*g;
    let correction=frame_store.neural.coefficients;
    let weight=frame_store.neural.training_weights[index]-0.003*(m/correction.x)/(sqrt(v/correction.y)+1e-8);
    frame_store.neural.training_weights[index]=weight;
    frame_store.neural.first_moment[index]=m;frame_store.neural.second_moment[index]=v;
    // Bias-corrected EMA is separate from the optimizer's training weights.
    frame_store.neural.weights[index]=(0.01*weight+0.99*correction.w*frame_store.neural.weights[index])/correction.z;
}
@compute @workgroup_size(1)
fn neural_finish_training() {
    var count=0u;for(var i=0u;i<NN_BATCH;i++) {count+=frame_store.neural.valid_samples[i];}
    frame_store.neural.state.x+=count;
    if count>0u {frame_store.neural.state.y++;}
    frame_store.neural.state.z++;
}
