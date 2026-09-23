#include <metal_stdlib>
using namespace metal;
struct Vertex { float4 position [[position]]; float2 uv; };
vertex Vertex vertexMain(uint index [[vertex_id]]) {
    const float2 positions[3] = {float2(-1,-1),float2(3,-1),float2(-1,3)};
    Vertex result; result.position = float4(positions[index],0,1);
    result.uv = float2((positions[index].x+1)*0.5,1-(positions[index].y+1)*0.5); return result;
}
fragment float4 fragmentMain(Vertex input [[stage_in]], texture2d<float> image [[texture(0)]]) {
    constexpr sampler filter(coord::normalized, address::clamp_to_edge, filter::linear);
    return image.sample(filter,input.uv);
}
kernel void baseline(device const float4* input [[buffer(0)]], texture2d<float, access::write> output [[texture(0)]],
                     uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x >= output.get_width() || pixel.y >= output.get_height()) return;
    output.write(float4(input[(pixel.y/2)*(output.get_width()/2)+pixel.x/2].xyz,1),pixel);
}