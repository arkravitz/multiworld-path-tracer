// Resource-free reference path, including neural teachers and shader probes.
const CAUSTICS_ENABLED: bool = false;
fn caustic_estimate(world:u32, position:vec3<f32>, normal:vec3<f32>, albedo:vec3<f32>, bounce:u32, camera_direction:vec3<f32>) -> vec3<f32> { return vec3<f32>(0.0); }

fn caustic_visible(index:u32, world:u32)->vec3<f32> {return vec3<f32>(0.0);}
