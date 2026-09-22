import Foundation

enum OptimizedShaders {
    static func header(_ precision: Precision) -> String {
        let scalar = precision == .f16 ? "half" : "float"
        return """
        #include <metal_stdlib>
        using namespace metal;
        #pragma STDC FP_CONTRACT OFF
        using Scalar = \(scalar);
        using Vec = \(scalar)4;
        struct OptShape { uint width, height, capture; };
        inline Vec feature(device const Vec* input, uint group, uint width, uint height, int column, int row) {
            if (column < 0 || row < 0 || column >= int(width) || row >= int(height)) return Vec(0);
            return input[group * width * height + uint(row) * width + uint(column)];
        }
        inline Vec rgb(device const float4* input, uint width, uint height, int column, int row) {
            if (column < 0 || row < 0 || column >= int(width) || row >= int(height)) return Vec(0);
            const float4 value = input[uint(row) * width + uint(column)];
            return Vec(Scalar(value.x), Scalar(value.y), Scalar(value.z), Scalar(0));
        }
        """
    }

    static func convolution(name: String, inputChannels: Int, kernel: Int, geometry: KernelGeometry,
                            tiled: Bool, activated: Bool, stem: Bool) -> String {
        let tileWidth = geometry.groupX * geometry.blockX + kernel - 1
        let tileHeight = geometry.groupY * geometry.blockY + kernel - 1
        let threads = geometry.groupX * geometry.groupY
        let radius = kernel / 2
        var lines = ["""
        kernel void \(name)(device const \(stem ? "float4" : "Vec")* input [[buffer(0)]],
            device const Vec* weights [[buffer(1)]], device const Scalar* biases [[buffer(2)]],
            device Vec* output [[buffer(3)]], constant OptShape& shape [[buffer(4)]],
            uint3 local [[thread_position_in_threadgroup]], uint linear [[thread_index_in_threadgroup]],
            uint3 groupPosition [[threadgroup_position_in_grid]]) {
            const uint width = shape.width, height = shape.height;
            const uint originX = groupPosition.x * \(geometry.groupX * geometry.blockX);
            const uint originY = groupPosition.y * \(geometry.groupY * geometry.blockY);
            const uint outX = originX + local.x * \(geometry.blockX);
            const uint outY = originY + local.y * \(geometry.blockY);
            const uint outputBase = groupPosition.z * \(geometry.outputBlock);
        """]
        if tiled { lines.append("threadgroup Vec tile[\(tileWidth * tileHeight)];") }
        for output in 0..<geometry.outputBlock { for row in 0..<geometry.blockY { for column in 0..<geometry.blockX {
            lines.append("Scalar acc_\(output)_\(row)_\(column) = biases[outputBase + \(output)];")
        } } }
        lines.append("for (uint inputGroup = 0; inputGroup < \((inputChannels + 3) / 4); ++inputGroup) {")
        if tiled {
            lines.append("""
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint element = linear; element < \(tileWidth * tileHeight); element += \(threads)) {
                const int column = int(originX) + int(element % \(tileWidth)) - \(radius);
                const int row = int(originY) + int(element / \(tileWidth)) - \(radius);
                tile[element] = feature(input, inputGroup, width, height, column, row);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            """)
        }
        lines.append("for (uint kernelRow = 0; kernelRow < \(kernel); ++kernelRow) { for (uint kernelColumn = 0; kernelColumn < \(kernel); ++kernelColumn) {")
        lines.append("const uint tap = inputGroup * \(kernel * kernel) + kernelRow * \(kernel) + kernelColumn;")
        for output in 0..<geometry.outputBlock {
            lines.append("const Vec weight_\(output) = weights[tap * 16 + outputBase + \(output)];")
        }
        for row in 0..<geometry.blockY { for column in 0..<geometry.blockX {
            let sample: String
            if tiled {
                sample = "tile[(local.y * \(geometry.blockY) + kernelRow + \(row)) * \(tileWidth) + local.x * \(geometry.blockX) + kernelColumn + \(column)]"
            } else {
                let coordinates = "int(outX) + \(column) + int(kernelColumn) - \(radius), int(outY) + \(row) + int(kernelRow) - \(radius)"
                sample = stem ? "rgb(input, width, height, \(coordinates))" : "feature(input, inputGroup, width, height, \(coordinates))"
            }
            lines.append("const Vec sample_\(row)_\(column) = \(sample);")
            for output in 0..<geometry.outputBlock {
                lines.append("acc_\(output)_\(row)_\(column) += dot(weight_\(output), sample_\(row)_\(column));")
            }
        } }
        lines.append("} } }")
        for row in 0..<geometry.blockY { for column in 0..<geometry.blockX {
            lines.append("if (outX + \(column) < width && outY + \(row) < height) {")
            for group in 0..<(geometry.outputBlock / 4) {
                let lanes = (0..<4).map { lane -> String in
                    let value = "acc_\(group * 4 + lane)_\(row)_\(column)"
                    return activated ? "tanh(\(value))" : value
                }.joined(separator: ", ")
                lines.append("output[(outputBase / 4 + \(group)) * width * height + (outY + \(row)) * width + outX + \(column)] = Vec(\(lanes));")
            }
            lines.append("}")
        } }
        lines.append("}")
        return lines.joined(separator: "\n")
    }

    static func head(_ candidate: MetalCandidate) -> String {
        let geometry = candidate.head
        var lines = ["""
        kernel void head(device const Vec* input [[buffer(0)]], device const Vec* weights [[buffer(1)]],
            device const Scalar* biases [[buffer(2)]], device float4* final [[buffer(3)]],
            constant OptShape& shape [[buffer(4)]], device const float4* original [[buffer(5)]],
            texture2d<float, access::write> output [[texture(0)]], uint2 position [[thread_position_in_grid]]) {
            const uint width = shape.width, height = shape.height, outWidth = width * 2, outHeight = height * 2;
            const uint outX = position.x * \(geometry.blockX), outY = position.y * \(geometry.blockY);
            if (outX >= outWidth || outY >= outHeight) return;
        """]
        for channel in 0..<3 { for row in 0..<geometry.blockY { for column in 0..<geometry.blockX {
            lines.append("Scalar acc_\(channel)_\(row)_\(column) = biases[\(channel)];")
        } } }
        lines.append("for (uint group = 0; group < 4; ++group) { for (int kernelRow = 0; kernelRow < 3; ++kernelRow) { for (int kernelColumn = 0; kernelColumn < 3; ++kernelColumn) {")
        for channel in 0..<3 { lines.append("const Vec weight_\(channel) = weights[(group * 9 + uint(kernelRow * 3 + kernelColumn)) * 3 + \(channel)];") }
        for row in 0..<geometry.blockY { for column in 0..<geometry.blockX {
            let tag = "\(row)_\(column)"
            lines.append("""
            const int column_\(tag) = int(outX) + \(column) + kernelColumn - 1;
            const int row_\(tag) = int(outY) + \(row) + kernelRow - 1;
            Vec sample_\(tag) = Vec(0);
            if (column_\(tag) >= 0 && row_\(tag) >= 0 && column_\(tag) < int(outWidth) && row_\(tag) < int(outHeight)) {
            """)
            let index = candidate.fusedHead
                ? "group * width * height + uint(row_\(tag)) / 2 * width + uint(column_\(tag)) / 2"
                : "group * outWidth * outHeight + uint(row_\(tag)) * outWidth + uint(column_\(tag))"
            lines.append("sample_\(tag) = input[\(index)]; }")
            for channel in 0..<3 { lines.append("acc_\(channel)_\(tag) += dot(weight_\(channel), sample_\(tag));") }
        } }
        lines.append("} } }")
        for row in 0..<geometry.blockY { for column in 0..<geometry.blockX {
            let tag = "\(row)_\(column)"
            lines.append("""
            if (outX + \(column) < outWidth && outY + \(row) < outHeight) {
                const uint column = outX + \(column), row = outY + \(row);
                const float4 value = float4(float(acc_0_\(tag)), float(acc_1_\(tag)), float(acc_2_\(tag)), 0.0f);
            """)
            if candidate.fusedHead {
                lines.append("""
                const float4 result = clamp(value + float4(original[(row / 2) * width + column / 2].xyz, 1.0f), 0.0f, 1.0f);
                if (shape.capture != 0) final[row * outWidth + column] = result;
                output.write(result, uint2(column, row));
                """)
            } else { lines.append("final[row * outWidth + column] = value;") }
            lines.append("}")
        } }
        lines.append("}")
        return lines.joined(separator: "\n")
    }

    static let utilities = """
    kernel void activate(device const Vec* input [[buffer(0)]], device Vec* output [[buffer(1)]],
        constant OptShape& shape [[buffer(2)]], uint index [[thread_position_in_grid]]) {
        if (index < shape.width * shape.height * 4) output[index] = tanh(input[index]);
    }
    kernel void nearest(device const Vec* input [[buffer(0)]], device Vec* output [[buffer(1)]],
        constant OptShape& shape [[buffer(2)]], uint index [[thread_position_in_grid]]) {
        const uint pixels = shape.width * shape.height * 4;
        if (index >= pixels * 4) return;
        const uint group = index / pixels, pixel = index % pixels;
        output[index] = input[group * shape.width * shape.height + (pixel / (shape.width * 2) / 2) * shape.width + pixel % (shape.width * 2) / 2];
    }
    kernel void compose(device const float4* head [[buffer(0)]], device const float4* original [[buffer(1)]],
        device float4* final [[buffer(2)]], constant OptShape& shape [[buffer(3)]],
        texture2d<float, access::write> output [[texture(0)]], uint index [[thread_position_in_grid]]) {
        const uint width = shape.width * 2;
        if (index >= width * shape.height * 2) return;
        const uint column = index % width, row = index / width;
        const float4 result = clamp(head[index] + float4(original[(row / 2) * shape.width + column / 2].xyz, 1.0f), 0.0f, 1.0f);
        if (shape.capture != 0) final[index] = result;
        output.write(result, uint2(column, row));
    }
    """

    static func source(candidate: MetalCandidate, precision: Precision) -> String {
        [header(precision), convolution(name: "stem", inputChannels: 3, kernel: 5, geometry: candidate.stem,
            tiled: false, activated: candidate.fusedActivation, stem: true),
         convolution(name: "body", inputChannels: 16, kernel: 3, geometry: candidate.body,
            tiled: candidate.tiled, activated: candidate.fusedActivation, stem: false), head(candidate), utilities].joined(separator: "\n")
    }
}