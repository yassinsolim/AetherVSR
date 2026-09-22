#include <metal_stdlib>
using namespace metal;
#pragma STDC FP_CONTRACT OFF

struct ConvolutionShape { uint width, height, inputChannels, outputChannels, kernelSize; };
struct TensorShape { uint width, height, channels; };

kernel void convolution32(device const float* input [[buffer(0)]],
                          device const float* weights [[buffer(1)]],
                          device const float* biases [[buffer(2)]],
                          device float* output [[buffer(3)]],
                          constant ConvolutionShape& shape [[buffer(4)]],
                          uint index [[thread_position_in_grid]]) {
    const uint pixels = shape.width * shape.height;
    if (index >= pixels * shape.outputChannels) return;
    const uint channel = index / pixels, pixel = index % pixels;
    const int column = int(pixel % shape.width), row = int(pixel / shape.width);
    const int radius = int(shape.kernelSize / 2);
    float value = biases[channel];
    for (uint inputChannel = 0; inputChannel < shape.inputChannels; ++inputChannel) {
        for (uint kernelRow = 0; kernelRow < shape.kernelSize; ++kernelRow) {
            for (uint kernelColumn = 0; kernelColumn < shape.kernelSize; ++kernelColumn) {
                const int sourceColumn = column + int(kernelColumn) - radius;
                const int sourceRow = row + int(kernelRow) - radius;
                if (sourceColumn < 0 || sourceRow < 0 || sourceColumn >= int(shape.width) || sourceRow >= int(shape.height)) continue;
                const uint weight = ((channel * shape.inputChannels + inputChannel) * shape.kernelSize + kernelRow) * shape.kernelSize + kernelColumn;
                value += input[inputChannel * pixels + uint(sourceRow) * shape.width + uint(sourceColumn)] * weights[weight];
            }
        }
    }
    output[index] = value;
}

kernel void activation32(device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
                         constant uint& count [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = tanh(input[index]);
}

kernel void nearest32(device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
                      constant TensorShape& shape [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    const uint outputWidth = shape.width * 2, outputPixels = shape.width * shape.height * 4;
    if (index >= outputPixels * shape.channels) return;
    const uint channel = index / outputPixels, pixel = index % outputPixels;
    output[index] = input[channel * shape.width * shape.height + (pixel / outputWidth / 2) * shape.width + (pixel % outputWidth / 2)];
}

kernel void residual32(device const float* head [[buffer(0)]], device const float* nearestInput [[buffer(1)]],
                       device float* output [[buffer(2)]], constant uint& count [[buffer(3)]],
                       uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = head[index] + nearestInput[index];
}

kernel void clamp32(device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
                    constant uint& count [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = clamp(input[index], 0.0f, 1.0f);
}

kernel void rgba32(device const float* input [[buffer(0)]], texture2d<float, access::write> output [[texture(0)]],
                   constant TensorShape& shape [[buffer(1)]], uint index [[thread_position_in_grid]]) {
    const uint pixels = shape.width * shape.height;
    if (index < pixels) output.write(float4(input[index], input[pixels + index], input[pixels * 2 + index], 1.0f),
                                    uint2(index % shape.width, index / shape.width));
}

kernel void convolution16(device const half* input [[buffer(0)]], device const half* weights [[buffer(1)]],
                          device const half* biases [[buffer(2)]], device half* output [[buffer(3)]],
                          constant ConvolutionShape& shape [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    const uint pixels = shape.width * shape.height;
    if (index >= pixels * shape.outputChannels) return;
    const uint channel = index / pixels, pixel = index % pixels;
    const int column = int(pixel % shape.width), row = int(pixel / shape.width), radius = int(shape.kernelSize / 2);
    half value = biases[channel];
    for (uint group = 0; group < (shape.inputChannels + 3) / 4; ++group) {
        for (uint kernelRow = 0; kernelRow < shape.kernelSize; ++kernelRow) {
            for (uint kernelColumn = 0; kernelColumn < shape.kernelSize; ++kernelColumn) {
                const int sourceColumn = column + int(kernelColumn) - radius, sourceRow = row + int(kernelRow) - radius;
                if (sourceColumn < 0 || sourceRow < 0 || sourceColumn >= int(shape.width) || sourceRow >= int(shape.height)) continue;
                half4 samples = half4(0), taps = half4(0);
                for (uint lane = 0; lane < 4; ++lane) {
                    const uint inputChannel = group * 4 + lane;
                    if (inputChannel < shape.inputChannels) {
                        samples[lane] = input[inputChannel * pixels + uint(sourceRow) * shape.width + uint(sourceColumn)];
                        taps[lane] = weights[((channel * shape.inputChannels + inputChannel) * shape.kernelSize + kernelRow) * shape.kernelSize + kernelColumn];
                    }
                }
                value += dot(samples, taps);
            }
        }
    }
    output[index] = value;
}

kernel void activation16(device const half* input [[buffer(0)]], device half* output [[buffer(1)]],
                         constant uint& count [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = tanh(input[index]);
}

kernel void nearest16(device const half* input [[buffer(0)]], device half* output [[buffer(1)]],
                      constant TensorShape& shape [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    const uint outputWidth = shape.width * 2, outputPixels = shape.width * shape.height * 4;
    if (index >= outputPixels * shape.channels) return;
    const uint channel = index / outputPixels, pixel = index % outputPixels;
    output[index] = input[channel * shape.width * shape.height + (pixel / outputWidth / 2) * shape.width + (pixel % outputWidth / 2)];
}

kernel void residual16(device const half* head [[buffer(0)]], device const float* nearestInput [[buffer(1)]],
                       device float* output [[buffer(2)]], constant uint& count [[buffer(3)]],
                       uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = float(head[index]) + nearestInput[index];
}