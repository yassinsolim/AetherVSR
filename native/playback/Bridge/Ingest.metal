#include <metal_stdlib>
using namespace metal;
#pragma STDC FP_CONTRACT OFF
struct ColorShape { uint width, height, format, centered; };
kernel void ingest(texture2d<float, access::read> luma [[texture(0)]],
                   texture2d<float, access::sample> chroma [[texture(1)]],
                   device float4* output [[buffer(0)]], constant ColorShape& shape [[buffer(1)]],
                   uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= shape.width || pixel.y >= shape.height) return;
    float3 rgb;
    if (shape.format == 0) rgb = luma.read(pixel).rgb;
    else {
        constexpr sampler linearClamp(coord::pixel, address::clamp_to_edge, filter::linear);
        const float2 coordinate = float2(float(pixel.x) * 0.5f + (shape.centered ? 0.25f : 0.5f), float(pixel.y) * 0.5f + 0.25f);
        const float2 uv = chroma.sample(linearClamp, coordinate).rg * 255.0f;
        const float y = (luma.read(pixel).r * 255.0f - (shape.format == 1 ? 16.0f : 0.0f)) / (shape.format == 1 ? 219.0f : 255.0f);
        const float cb = (uv.x - 128.0f) / (shape.format == 1 ? 224.0f : 255.0f);
        const float cr = (uv.y - 128.0f) / (shape.format == 1 ? 224.0f : 255.0f);
        rgb = float3(y + 1.5748f * cr, y - 0.1873242729306488f * cb - 0.4681242729306488f * cr, y + 1.8556f * cb);
    }
    output[pixel.y * shape.width + pixel.x] = float4(clamp(rgb, 0.0f, 1.0f), 0.0f);
}