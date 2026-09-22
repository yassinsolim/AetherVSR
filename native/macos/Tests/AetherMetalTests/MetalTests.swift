import Foundation
import Metal
import XCTest
@testable import AetherMetal

final class MetalTests: XCTestCase {
    func testF32GoldenAndLifecycle() throws {
        guard ProcessInfo.processInfo.environment["AETHERVSR_METAL_TESTS"] == "1" else { throw XCTSkip("Physical Metal test is opt-in, not CI hardware evidence") }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let model = try ProductionModel(data: Data(contentsOf: root.appendingPathComponent("public/models/aethersr-c16d2.json")))
        let golden = try Golden.load(Data(contentsOf: root.appendingPathComponent("public/models/golden-c16d2.json")))
        let engine = try MetalEngine(model: model, extent: Extent(width: golden.width, height: golden.height), precision: .f32)
        defer { engine.destroy() }
        XCTAssertThrowsError(try engine.capture())
        XCTAssertThrowsError(try engine.loadInput([0]))
        try engine.loadInput(golden.input)
        let actual = try engine.capture()
        let comparisons = try Validation.qualify(actual.checkpoints, expected: golden.expected, precision: .f32, declaredPrecision: "f32")
        for result in comparisons { XCTAssertTrue(result.passed, "\(result.stage): \(String(describing: result.maxAbsError))") }
        XCTAssertEqual(actual.rgba.count, golden.width * golden.height * 16)
        XCTAssertTrue(try Validation.rgba(actual.rgba, expected: golden.expected.last!, precision: .f32).passed)
        for value in [Float.nan, Float.infinity, -Float.infinity, Float(10)] {
            var corrupt = actual.checkpoints; corrupt[2].values[19] = value
            XCTAssertFalse(try Validation.qualify(corrupt, expected: golden.expected, precision: .f32, declaredPrecision: "f32").allSatisfy(\.passed))
        }
        var permuted = actual.checkpoints
        permuted[0].values = Array(permuted[0].values.reversed())
        XCTAssertFalse(try Validation.qualify(permuted, expected: golden.expected, precision: .f32, declaredPrecision: "f32").allSatisfy(\.passed))
        engine.destroy(); engine.destroy()
        XCTAssertThrowsError(try engine.capture())
        XCTAssertThrowsError(try engine.loadInput(golden.input))
    }

    func testAsymmetricConvolutionLayout() throws {
        guard ProcessInfo.processInfo.environment["AETHERVSR_METAL_TESTS"] == "1" else { throw XCTSkip("Physical Metal test is opt-in") }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { return XCTFail("Metal unavailable") }
        let options = MTLCompileOptions(); options.languageVersion = .version3_0; options.fastMathEnabled = false
        let library = try device.makeLibrary(source: MetalEngine.shaderSource(), options: options)
        let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "convolution32")!)
        let width = 3, height = 2, inputs = 2, outputs = 2, kernelSize = 3, pixels = width * height
        let input: [Float] = [1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60]
        let weights: [Float] = (0..<(outputs * inputs * 9)).map { Float($0 + 1) / 64 }
        let biases: [Float] = [0.25, -0.5]
        var expected = [Float](repeating: 0, count: pixels * outputs)
        for output in 0..<outputs {
            for row in 0..<height {
                for column in 0..<width {
                    var value = biases[output]
                    for inputChannel in 0..<inputs {
                        for kernelRow in 0..<kernelSize {
                            for kernelColumn in 0..<kernelSize {
                                let sourceRow = row + kernelRow - 1, sourceColumn = column + kernelColumn - 1
                                if sourceRow >= 0 && sourceColumn >= 0 && sourceRow < height && sourceColumn < width {
                                    value += input[inputChannel * pixels + sourceRow * width + sourceColumn] *
                                        weights[((output * inputs + inputChannel) * kernelSize + kernelRow) * kernelSize + kernelColumn]
                                }
                            }
                        }
                    }
                    expected[output * pixels + row * width + column] = value
                }
            }
        }
        func buffer(_ values: [Float]) -> MTLBuffer {
            values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
        }
        let data = [buffer(input), buffer(weights), buffer(biases), buffer([Float](repeating: .nan, count: pixels * outputs))]
        let dimensions = [UInt32(width), UInt32(height), UInt32(inputs), UInt32(outputs), UInt32(kernelSize)]
        let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in data.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
        dimensions.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 4) }
        encoder.dispatchThreads(MTLSize(width: pixels * outputs, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed); XCTAssertNil(command.error)
        let actual = Array(UnsafeBufferPointer(start: data[3].contents().assumingMemoryBound(to: Float.self), count: pixels * outputs))
        XCTAssertEqual(actual, expected)
        XCTAssertNotEqual(actual[0], actual[1]); XCTAssertNotEqual(actual[0], actual[pixels])
    }
}