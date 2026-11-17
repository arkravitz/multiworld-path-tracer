#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
constant uint H = @WIDTH@;

// Packed header: count, width, weights, masks, features, predictions, clamp, spare.
// All bindings belong to this small standalone wgpu compute pipeline.
kernel void tensor_inference(device float* data [[buffer(0)]],
                             uint lane [[thread_index_in_threadgroup]],
                             uint2 group [[threadgroup_position_in_grid]]) {
    device uint* cfg=reinterpret_cast<device uint*>(data);
    uint first=(group.x+group.y*65535u)*32u;
    if (first>=cfg[0]) return;
    threadgroup half a[32*H];
    threadgroup half b[H*H];
    threadgroup float c[32*H];
    threadgroup uint live[128];
    live[lane]=lane<32 && first+lane<cfg[0] && data[cfg[3]+first+lane]>0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint active=0;
    for(uint i=0;i<32;i++) active+=live[i];
    if(active==0) return;
    for(uint i=lane;i<32*H;i+=128) {
        uint q=first+i/H;
        a[i]=half(q<cfg[0] && data[cfg[3]+q]>0.0f?data[cfg[4]+q*H+i%H]:0.0f);
    }
    constexpr auto desc=matmul2d_descriptor(32,H,H,false,false,false);
    matmul2d<desc,execution_simdgroups<4>> op;
    for(uint layer=0;layer<2;layer++) {
        uint weights=cfg[2]+layer*(H*H+H);
        for(uint i=lane;i<H*H;i+=128) b[i]=half(data[weights+i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto ta=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(a,dextents<int32_t,2>(H,32));
        auto tb=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(b,dextents<int32_t,2>(H,H));
        auto tc=op.get_destination_cooperative_tensor<decltype(ta),decltype(tb),float>();
        op.run(ta,tb,tc);
        #pragma unroll
        for(uint16_t i=0;i<tc.get_capacity();i++) {
            auto index=tc.get_multidimensional_index(i);
            if(index[0]<H && index[1]<32) {
                tc[i]=max(0.0f,tc[i]+data[weights+H*H+index[0]]);
            }
        }
        auto target=tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(c,dextents<int32_t,2>(H,32));
        tc.store(target);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(layer==0) {
            for(uint i=lane;i<32*H;i+=128) a[i]=half(c[i]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if(lane<32 && first+lane<cfg[0]) {
        for(uint channel=0;channel<3;channel++) {
            uint weights=cfg[2]+2*H*H+2*H;
            float sum=data[weights+3*H+channel];
            for(uint j=0;j<H;j++) sum+=c[lane*H+j]*data[weights+j*3+channel];
            data[cfg[5]+(first+lane)*3+channel]=clamp(sum,0.0f,as_type<float>(cfg[6]));
        }
    }
}
