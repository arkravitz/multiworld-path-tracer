// Separate bind group: denoising and query resolution use it in different passes.
@group(1) @binding(3) var<storage,read_write> neural_queries:array<NeuralSample>;
fn neural_enqueue(index:u32,sample:NeuralSample) {neural_queries[index]=sample;}
fn neural_clear_queries(index:u32) {
    neural_queries[index*select(1u,2u,ACTIVE_MODE==2u)].label=vec4<f32>(0.0);
    if ACTIVE_MODE==2u {neural_queries[index*2u+1u].label=vec4<f32>(0.0);}
}
