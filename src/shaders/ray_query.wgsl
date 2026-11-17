@group(0) @binding(8) var hardware_a: acceleration_structure;
@group(0) @binding(9) var hardware_b: acceleration_structure;

// Keep exact analytic intersections, shading, unbounded planes and scene-index
// tie breaking. Nonopaque triangles expose candidates before commitment too.
fn intersect_hardware(world:u32,ro:vec3<f32>,rd:vec3<f32>,glass_only:bool)->Hit {
    // Our box reference treats nonzero components below 1e-8 as parallel.
    // Native bounds use the real ray, which can diverge over a long distance.
    // Preserve that established convention for these rare rays.
    if any((abs(rd)>vec3<f32>(0.0)) & (abs(rd)<vec3<f32>(1e-8))) {
        if glass_only {return glass_boundary_software(world,ro,rd);}
        return intersect_bvh(world,ro,rd);
    }
    var nearest=1e30; var nearest_id=0xffffffffu;
    let planes=select(params.acceleration.z,params.acceleration.w,world==1u);
    let metadata=select(params.acceleration.x+params.lights.x,params.acceleration.y+params.lights.y,world==1u);
    for(var i=0u;i<planes;i++) {
        let index=node_at(world,i).object;
        let o=object_at(world,index);
        if glass_only && o.color.w<2.0 {continue;}
        let t=primitive_distance(o,ro,rd);
        if t<nearest || (t<1e30 && t==nearest && index<nearest_id) {nearest=t;nearest_id=index;}
    }
    var query:ray_query;
    let ray=RayDesc(RAY_FLAG_FORCE_NO_OPAQUE,255u,0.0,1e30,ro,rd);
    if world==0u {rayQueryInitialize(&query,hardware_a,ray);}
    else {rayQueryInitialize(&query,hardware_b,ray);}
    while rayQueryProceed(&query) {
        let candidate=rayQueryGetCandidateIntersection(&query);
        // Each source triangle has eight conservative native prism facets.
        let primitive=select(candidate.primitive_index,candidate.primitive_index/8u,candidate.kind==RAY_QUERY_INTERSECTION_TRIANGLE);
        let index=node_at(world,metadata+candidate.instance_custom_data+primitive).object;
        let o=object_at(world,index);
        if glass_only && o.color.w<2.0 {continue;}
        let t=primitive_distance(o,ro,rd);
        if t<nearest || (t<1e30 && t==nearest && index<nearest_id) {nearest=t;nearest_id=index;}
    }
    if nearest_id!=0xffffffffu {return primitive(object_at(world,nearest_id),ro,rd);}
    return Hit(1e30,vec3<f32>(0.0),vec3<f32>(0.0),vec3<f32>(0.0),0.0,0.0,0.0);
}
