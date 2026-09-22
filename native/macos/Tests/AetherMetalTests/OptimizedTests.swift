import Foundation
import Metal
import XCTest
@testable import AetherMetal

final class OptimizedTests: XCTestCase {
    private var root: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }
    private func data(_ name: String) throws -> Data { try Data(contentsOf: root.appendingPathComponent("public/models/\(name)")) }

    func testPackedLayoutsAndCorruption() throws {
        for channels in [3, 8, 16] {
            let extent = try Extent(width: 7, height: 3)
            let planar = (0..<(channels * extent.pixels)).map { Float($0 + 1) / 64 }
            let packed = try PackedLayout.pack(planar, extent: extent, channels: channels)
            XCTAssertEqual(try PackedLayout.unpack(packed, extent: extent, channels: channels), planar)
            for group in 0..<((channels + 3) / 4) { for pixel in 0..<extent.pixels { for lane in 0..<4 {
                let channel = group * 4 + lane
                XCTAssertEqual(packed[(group * extent.pixels + pixel) * 4 + lane], channel < channels ? planar[channel * extent.pixels + pixel] : 0)
            } } }
            var corrupt = packed; corrupt.swapAt(0, 1)
            XCTAssertNotEqual(try PackedLayout.unpack(corrupt, extent: extent, channels: channels), planar)
            XCTAssertThrowsError(try PackedLayout.unpack(Array(packed.dropLast()), extent: extent, channels: channels))
            for outputs in [3, 16] { for kernel in [3, 5] {
                let source = (0..<(outputs * channels * kernel * kernel)).map { Float($0) / 8192 }
                let weights = try PackedLayout.weights(source, inputChannels: channels, outputChannels: outputs, kernel: kernel)
                for output in 0..<outputs { for channel in 0..<channels { for tap in 0..<(kernel * kernel) {
                    let index = (((channel / 4) * kernel * kernel + tap) * outputs + output) * 4 + channel % 4
                    XCTAssertEqual(weights[index], source[(output * channels + channel) * kernel * kernel + tap])
                } } }
            } }
        }
        let model = try ProductionModel(data: data("aethersr-c16d2.json"))
        for precision in Precision.allCases {
            let packed = try PackedParameters(model: model, precision: precision)
            XCTAssertEqual(packed.hashes, try PackedParameters(model: model, precision: precision).hashes)
            try packed.verify(packed.buffers, modelHash: ProductionModel.fileHash)
            XCTAssertThrowsError(try packed.verify(packed.buffers, modelHash: "wrong"))
            var corrupt = packed.buffers; corrupt["body.0.weight"]![0] ^= 1
            XCTAssertThrowsError(try packed.verify(corrupt, modelHash: ProductionModel.fileHash))
            corrupt = packed.buffers; corrupt.removeValue(forKey: "stem.bias")
            XCTAssertThrowsError(try packed.verify(corrupt, modelHash: ProductionModel.fileHash))
            XCTAssertThrowsError(try PackedLayout.bytes([.nan], precision: precision))
            XCTAssertThrowsError(try PackedLayout.bytes([.infinity], precision: precision))
        }
        XCTAssertThrowsError(try PackedLayout.bytes([Float.greatestFiniteMagnitude], precision: .f16))
        XCTAssertEqual(MetalCandidate.allCases.map(\.rawValue), ["B", "C", "D", "E", "F", "G"])
    }

    private func device() throws -> MTLDevice {
        guard ProcessInfo.processInfo.environment["AETHERVSR_METAL_TESTS"] == "1" else { throw XCTSkip("Physical optimized Metal tests are opt-in") }
        guard let device = MTLCreateSystemDefaultDevice() else { throw QualificationError("Metal unavailable") }
        return device
    }

    private func library(_ device: MTLDevice, candidate: MetalCandidate, precision: Precision) throws -> MTLLibrary {
        let options = MTLCompileOptions(); options.languageVersion = .version3_0; options.fastMathEnabled = false
        return try device.makeLibrary(source: OptimizedShaders.source(candidate: candidate, precision: precision), options: options)
    }

    private func buffer(_ device: MTLDevice, _ bytes: Data) -> MTLBuffer {
        bytes.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
    }

    private func values(_ buffer: MTLBuffer, precision: Precision) -> [Float] {
        if precision == .f16 {
            return UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float16.self), count: buffer.length / 2).map { Float($0) }
        }
        return Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: buffer.length / 4))
    }

    private func convolution(device: MTLDevice, library: MTLLibrary, candidate: MetalCandidate, precision: Precision,
                             input: [Float], weights: [Float], biases: [Float], extent: Extent, stem: Bool) throws -> [Float] {
        let channels = stem ? 3 : 16, kernel = stem ? 5 : 3, geometry = stem ? candidate.stem : candidate.body
        let packedInput = try PackedLayout.pack(input, extent: extent, channels: channels)
        let source = buffer(device, try PackedLayout.bytes(packedInput, precision: stem ? .f32 : precision))
        let weight = buffer(device, try PackedLayout.bytes(PackedLayout.weights(weights, inputChannels: channels, outputChannels: 16, kernel: kernel), precision: precision))
        let bias = buffer(device, try PackedLayout.bytes(biases, precision: precision))
        let output = device.makeBuffer(length: extent.pixels * 16 * precision.bytesPerElement, options: .storageModeShared)!
        memset(output.contents(), 0xff, output.length)
        let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: stem ? "stem" : "body")!)
        XCTAssertLessThanOrEqual(pipeline.staticThreadgroupMemoryLength, device.maxThreadgroupMemoryLength)
        XCTAssertGreaterThanOrEqual(pipeline.maxTotalThreadsPerThreadgroup, geometry.groupX * geometry.groupY)
        let dimensions: [UInt32] = [UInt32(extent.width), UInt32(extent.height), 1]
        let command = device.makeCommandQueue()!.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        for (index, resource) in [source, weight, bias, output].enumerated() { encoder.setBuffer(resource, offset: 0, index: index) }
        dimensions.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 4) }
        let tileWidth = geometry.groupX * geometry.blockX, tileHeight = geometry.groupY * geometry.blockY
        encoder.dispatchThreadgroups(MTLSize(width: (extent.width + tileWidth - 1) / tileWidth,
            height: (extent.height + tileHeight - 1) / tileHeight, depth: 16 / geometry.outputBlock),
            threadsPerThreadgroup: MTLSize(width: geometry.groupX, height: geometry.groupY, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed); XCTAssertNil(command.error)
        return try PackedLayout.unpack(values(output, precision: precision), extent: extent, channels: 16)
    }

    private func oracle(input: [Float], weights: [Float], biases: [Float], extent: Extent, channels: Int,
                        kernel: Int, precision: Precision, activation: Bool) -> [Float] {
        func round(_ value: Float) -> Float { precision == .f16 ? Float(Float16(value)) : value }
        var result = [Float](repeating: 0, count: extent.pixels * 16)
        for output in 0..<16 { for row in 0..<extent.height { for column in 0..<extent.width {
            var value = round(biases[output])
            for group in 0..<((channels + 3) / 4) { for kernelRow in 0..<kernel { for kernelColumn in 0..<kernel {
                let sourceRow = row + kernelRow - kernel / 2, sourceColumn = column + kernelColumn - kernel / 2
                if sourceRow < 0 || sourceRow >= extent.height || sourceColumn < 0 || sourceColumn >= extent.width { continue }
                var dot: Float = 0
                for lane in 0..<4 where group * 4 + lane < channels {
                    let channel = group * 4 + lane
                    let sample = round(input[channel * extent.pixels + sourceRow * extent.width + sourceColumn])
                    let tap = round(weights[((output * channels + channel) * kernel + kernelRow) * kernel + kernelColumn])
                    dot = round(dot + round(sample * tap))
                }
                value = round(value + dot)
            } } }
            result[output * extent.pixels + row * extent.width + column] = activation ? round(tanh(value)) : value
        } } }
        return result
    }

    func testTinyConvolutionsAndRounding() throws {
        let device = try device(), extent = try Extent(width: 19, height: 11)
        for candidate in MetalCandidate.allCases { for precision in Precision.allCases {
            let library = try library(device, candidate: candidate, precision: precision)
            for stem in [true, false] {
                let channels = stem ? 3 : 16, kernel = stem ? 5 : 3
                let input = (0..<(channels * extent.pixels)).map { index -> Float in
                    let channel = index / extent.pixels, pixel = index % extent.pixels
                    let value = ((pixel / extent.width) * 3 + (pixel % extent.width) * 7 + channel * 11) % 17 - 8
                    return Float(value) / 32
                }
                let weights = (0..<(16 * channels * kernel * kernel)).map { Float(($0 * 11 + $0 / (channels * kernel * kernel)) % 7 - 3) / 64 }
                let biases = (0..<16).map { Float($0 % 5 - 2) / 16 }
                let expected = oracle(input: input, weights: weights, biases: biases, extent: extent, channels: channels,
                                      kernel: kernel, precision: precision, activation: candidate.fusedActivation)
                let actual = try convolution(device: device, library: library, candidate: candidate, precision: precision,
                    input: input, weights: weights, biases: biases, extent: extent, stem: stem)
                XCTAssertEqual(actual.count, expected.count)
                for index in actual.indices {
                    XCTAssertTrue(actual[index].isFinite)
                    XCTAssertEqual(actual[index], expected[index], accuracy: precision == .f16 ? 0.0002442 : 0.000001,
                                   "\(candidate.rawValue) \(precision) stem=\(stem) index=\(index)")
                }
                let permuted = (0..<channels).reversed().flatMap { Array(input[($0 * extent.pixels)..<(($0 + 1) * extent.pixels)]) }
                XCTAssertNotEqual(expected, oracle(input: permuted, weights: weights, biases: biases, extent: extent,
                    channels: channels, kernel: kernel, precision: precision, activation: candidate.fusedActivation))
                let spatiallyPermuted = (0..<channels).flatMap { Array(input[($0 * extent.pixels)..<(($0 + 1) * extent.pixels)].reversed()) }
                XCTAssertNotEqual(expected, oracle(input: spatiallyPermuted, weights: weights, biases: biases, extent: extent,
                    channels: channels, kernel: kernel, precision: precision, activation: candidate.fusedActivation))
            }
            if precision == .f16 {
                let tiny = try Extent(width: 1, height: 1)
                for sparse in [false, true] {
                    var weights = [Float](repeating: 0, count: 16 * 16 * 9)
                    for output in 0..<16 { for channel in 0..<16 {
                        if sparse ? (channel < 12 && channel % 4 == 0) : channel < 8 { weights[(output * 16 + channel) * 9 + 4] = 1 / 2048 }
                    } }
                    let actual = try convolution(device: device, library: library, candidate: candidate, precision: .f16,
                        input: [Float](repeating: 1, count: 16), weights: weights, biases: [Float](repeating: 1, count: 16), extent: tiny, stem: false)
                    let value: Float = sparse ? 1 : 1 + 1 / 256
                    let expected = candidate.fusedActivation ? Float(Float16(tanh(value))) : value
                    let wrong: Float = sparse ? Float(Float16(1 + Float(3) / 2048)) : 1
                    let wrongActivated = candidate.fusedActivation ? Float(Float16(tanh(wrong))) : wrong
                    XCTAssertNotEqual(expected, wrongActivated)
                    XCTAssertTrue(actual.allSatisfy { abs($0 - expected) <= 0.0004883 && abs($0 - expected) < abs($0 - wrongActivated) })
                }
            }
        } }
    }

    func testTinyHeadEdgesAndComposition() throws {
        let device = try device(), extent = try Extent(width: 19, height: 11)
        let outputPixels = extent.pixels * 4, outputWidth = extent.width * 2, outputHeight = extent.height * 2
        let planar = (0..<(extent.pixels * 16)).map { Float(($0 * 7 + $0 / extent.pixels) % 5 - 2) / 16 }
        let original = (0..<(extent.pixels * 3)).map { Float(($0 * 11) % 29) / 31 }
        let weights = (0..<432).map { index -> Float in
            let output = index / 144
            let value = (index * 5 + output) % 7 - 3
            return Float(value) / 64
        }
        let biases: [Float] = [-0.0625, 0, 0.0625]
        for precision in Precision.allCases {
            func round(_ value: Float) -> Float { precision == .f16 ? Float(Float16(value)) : value }
            var expected = [Float](repeating: 0, count: outputPixels * 3)
            for output in 0..<3 { for row in 0..<outputHeight { for column in 0..<outputWidth {
                var value = round(biases[output])
                for group in 0..<4 { for kernelRow in 0..<3 { for kernelColumn in 0..<3 {
                    let sourceRow = row + kernelRow - 1, sourceColumn = column + kernelColumn - 1
                    if sourceRow < 0 || sourceRow >= outputHeight || sourceColumn < 0 || sourceColumn >= outputWidth { continue }
                    var dot: Float = 0
                    for lane in 0..<4 {
                        let channel = group * 4 + lane
                        let sample = round(planar[channel * extent.pixels + (sourceRow / 2) * extent.width + sourceColumn / 2])
                        let tap = round(weights[(output * 16 + channel) * 9 + kernelRow * 3 + kernelColumn])
                        dot = round(dot + round(sample * tap))
                    }
                    value = round(value + dot)
                } } }
                value += original[output * extent.pixels + (row / 2) * extent.width + column / 2]
                expected[output * outputPixels + row * outputWidth + column] = min(1, max(0, value))
            } } }
            XCTAssertNotEqual(expected[2 * outputWidth + 2], expected[2 * outputWidth + 3])
            for candidate in MetalCandidate.allCases {
                let (actual, rgba) = try headOutput(device: device, candidate: candidate, precision: precision,
                    planar: planar, original: original, weights: weights, biases: biases, extent: extent)
                for channel in 0..<3 { for pixel in 0..<outputPixels {
                    XCTAssertEqual(actual[pixel * 4 + channel], expected[channel * outputPixels + pixel], accuracy: 0.000001, "\(candidate) \(precision) pixel \(pixel)")
                } }
                XCTAssertTrue(try Validation.rgba(rgba, expected: StageTensor(name: "final", width: outputWidth, height: outputHeight,
                    channels: 3, values: expected), precision: .f32).passed)
            }
        }
    }

    private func headOutput(device: MTLDevice, candidate: MetalCandidate, precision: Precision, planar: [Float], original: [Float],
                            weights: [Float], biases: [Float], extent: Extent) throws -> ([Float], Data) {
                let outputPixels = extent.pixels * 4, outputWidth = extent.width * 2, outputHeight = extent.height * 2
                let library = try library(device, candidate: candidate, precision: precision)
                var source = try PackedLayout.pack(planar, extent: extent, channels: 16)
                if !candidate.fusedHead {
                    var nearest = [Float](repeating: 0, count: source.count * 4)
                    for group in 0..<4 { for pixel in 0..<outputPixels { for lane in 0..<4 {
                        let originalPixel = (pixel / outputWidth / 2) * extent.width + pixel % outputWidth / 2
                        nearest[(group * outputPixels + pixel) * 4 + lane] = source[(group * extent.pixels + originalPixel) * 4 + lane]
                    } } }
                    source = nearest
                }
                let inputs = [buffer(device, try PackedLayout.bytes(source, precision: precision)),
                    buffer(device, try PackedLayout.bytes(PackedLayout.weights(weights, inputChannels: 16, outputChannels: 3, kernel: 3), precision: precision)),
                    buffer(device, try PackedLayout.bytes(biases, precision: precision))]
                let originalBuffer = buffer(device, try PackedLayout.bytes(PackedLayout.pack(original, extent: extent, channels: 3), precision: .f32))
                let final = device.makeBuffer(length: outputPixels * 16, options: .storageModeShared)!
                let head = device.makeBuffer(length: outputPixels * 16, options: .storageModeShared)!
                memset(final.contents(), 0xff, final.length)
                let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: outputWidth, height: outputHeight, mipmapped: false)
                descriptor.storageMode = .shared; descriptor.usage = [.shaderWrite, .shaderRead]
                let texture = device.makeTexture(descriptor: descriptor)!
                let dimensions: [UInt32] = [UInt32(extent.width), UInt32(extent.height), 1]
                let command = device.makeCommandQueue()!.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
                let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "head")!)
                encoder.setComputePipelineState(pipeline)
                for (index, resource) in (inputs + [candidate.fusedHead ? final : head]).enumerated() { encoder.setBuffer(resource, offset: 0, index: index) }
                dimensions.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 4) }
                encoder.setBuffer(originalBuffer, offset: 0, index: 5); encoder.setTexture(texture, index: 0)
                let geometry = candidate.head
                encoder.dispatchThreadgroups(MTLSize(width: (outputWidth + geometry.groupX * geometry.blockX - 1) / (geometry.groupX * geometry.blockX),
                    height: (outputHeight + geometry.groupY * geometry.blockY - 1) / (geometry.groupY * geometry.blockY), depth: 1),
                    threadsPerThreadgroup: MTLSize(width: geometry.groupX, height: geometry.groupY, depth: 1))
                encoder.endEncoding()
                if !candidate.fusedHead {
                    let compose = command.makeComputeCommandEncoder()!
                    compose.setComputePipelineState(try device.makeComputePipelineState(function: library.makeFunction(name: "compose")!))
                    for (index, resource) in [head, originalBuffer, final].enumerated() { compose.setBuffer(resource, offset: 0, index: index) }
                    dimensions.withUnsafeBytes { compose.setBytes($0.baseAddress!, length: $0.count, index: 3) }
                    compose.setTexture(texture, index: 0)
                    compose.dispatchThreads(MTLSize(width: outputPixels, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                    compose.endEncoding()
                }
                command.commit(); command.waitUntilCompleted(); XCTAssertEqual(command.status, .completed); XCTAssertNil(command.error)
                let actual = values(final, precision: .f32)
                var rgba = [UInt8](repeating: 0, count: outputPixels * 4)
                rgba.withUnsafeMutableBytes { texture.getBytes($0.baseAddress!, bytesPerRow: outputWidth * 4,
                    from: MTLRegionMake2D(0, 0, outputWidth, outputHeight), mipmapLevel: 0) }
                return (actual, Data(rgba))
    }

    func testStemAndHeadRoundingControls() throws {
        let device = try device(), stemExtent = try Extent(width: 3, height: 3), headExtent = try Extent(width: 1, height: 1)
        for candidate in MetalCandidate.allCases {
            let library = try library(device, candidate: candidate, precision: .f16)
            for sparse in [false, true] {
                var stemWeights = [Float](repeating: 0, count: 16 * 3 * 25)
                for output in 0..<16 {
                    if sparse {
                        for tap in [11, 12, 13] { stemWeights[output * 75 + tap] = 1 / 2048 }
                    } else {
                        for channel in 0..<3 { stemWeights[(output * 3 + channel) * 25 + 12] = 1 / 2048 }
                    }
                }
                let stem = try convolution(device: device, library: library, candidate: candidate, precision: .f16,
                    input: [Float](repeating: 1, count: 27), weights: stemWeights, biases: [Float](repeating: 1, count: 16), extent: stemExtent, stem: true)
                let grouped = Float(Float16(1 + Float(3) / 2048))
                let correct: Float = sparse ? 1 : grouped, incorrect: Float = sparse ? grouped : 1
                let expected = candidate.fusedActivation ? Float(Float16(tanh(correct))) : correct
                let wrong = candidate.fusedActivation ? Float(Float16(tanh(incorrect))) : incorrect
                XCTAssertNotEqual(expected, wrong)
                for output in 0..<16 { XCTAssertLessThan(abs(stem[output * 9 + 4] - expected), abs(stem[output * 9 + 4] - wrong)) }

                var headWeights = [Float](repeating: 0, count: 432)
                for output in 0..<3 { for channel in 0..<16 {
                    if sparse ? (channel < 12 && channel % 4 == 0) : channel < 8 { headWeights[(output * 16 + channel) * 9 + 4] = 1 / 4096 }
                } }
                let rgb: [Float] = [0.10001, 0.20003, 0.30007]
                let (head, _) = try headOutput(device: device, candidate: candidate, precision: .f16,
                    planar: [Float](repeating: 1, count: 16), original: rgb, weights: headWeights, biases: [0.5, 0.5, 0.5], extent: headExtent)
                let halfValue: Float = sparse ? 0.5 : 0.5 + Float(1) / 512
                let wrongValue: Float = sparse ? Float(Float16(0.5 + Float(3) / 4096)) : 0.5
                for channel in 0..<3 {
                    let expectedValue = halfValue + rgb[channel]
                    XCTAssertEqual(head[channel], expectedValue)
                    XCTAssertNotEqual(head[channel], wrongValue + rgb[channel])
                    XCTAssertNotEqual(head[channel], halfValue + Float(Float16(rgb[channel])))
                    XCTAssertNotEqual(head[channel], Float(Float16(halfValue + Float(Float16(rgb[channel])))))
                }
            }
        }
    }

    func testCandidateGoldenAndFastParity() throws {
        _ = try device()
        let model = try ProductionModel(data: data("aethersr-c16d2.json")), golden = try Golden.load(data("golden-c16d2.json"))
        let extent = try Extent(width: golden.width, height: golden.height)
        for candidate in MetalCandidate.allCases { for precision in Precision.allCases {
            let diagnostic = try OptimizedEngine(model: model, extent: extent, precision: precision, candidate: candidate, diagnostic: true)
            let fast = try OptimizedEngine(model: model, extent: extent, precision: precision, candidate: candidate, diagnostic: false)
            XCTAssertThrowsError(try diagnostic.capture()); try diagnostic.loadInput(golden.input); try fast.loadInput(golden.input)
            let actual = try diagnostic.capture(), fastOutput = try fast.capture()
            let comparisons = try Validation.qualify(actual.checkpoints, expected: golden.expected, precision: precision, declaredPrecision: precision.rawValue)
            XCTAssertTrue(comparisons.allSatisfy(\.passed), "\(candidate) \(precision): \(comparisons)")
            XCTAssertTrue(try Validation.rgba(actual.rgba, expected: golden.expected.last!, precision: precision).passed)
            XCTAssertEqual(fastOutput.rgba, actual.rgba)
            for poison in [Float.nan, Float.infinity, -Float.infinity, 10] {
                var corrupt = actual.checkpoints; corrupt[1].values[7] = poison
                XCTAssertFalse(try Validation.qualify(corrupt, expected: golden.expected, precision: precision, declaredPrecision: precision.rawValue).allSatisfy(\.passed))
            }
            diagnostic.destroy(); diagnostic.destroy(); fast.destroy()
            XCTAssertThrowsError(try diagnostic.capture()); XCTAssertThrowsError(try diagnostic.loadInput(golden.input))
        } }
    }
}