// Parser-complete fallback for devices without cooperative matrix support.
// The call is specialized away when NRC_COOPERATIVE is false.
fn neural_resolve_matrix(lane:u32)->vec3<f32> {return vec3<f32>(0.0);}

fn neural_enqueue(index:u32,sample:NeuralSample) {}
fn neural_clear_queries(index:u32) {}
