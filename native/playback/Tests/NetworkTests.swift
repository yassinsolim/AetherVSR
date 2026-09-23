import AetherMetal
import Foundation
import Metal
import XCTest

final class NetworkTests: XCTestCase {
    func testFrozenFCommandAdapter() throws {
        guard ProcessInfo.processInfo.environment["AETHERVSR_METAL_TESTS"] == "1" else { throw XCTSkip("Physical candidate-F integration test is opt-in") }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let model = try ProductionModel(data: Data(contentsOf: root.appendingPathComponent("public/models/aethersr-c16d2.json")))
        let golden = try Golden.load(Data(contentsOf: root.appendingPathComponent("public/models/golden-c16d2.json")))
        let extent = try Extent(width: golden.width, height: golden.height), device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        for precision in Precision.allCases {
            let frame = try FrameNetwork(device: device, model: model, extent: extent, precision: precision, diagnostic: true)
            let original = try OptimizedEngine(model: model, extent: extent, precision: precision, candidate: .f, diagnostic: true)
            defer { original.destroy() }
            XCTAssertEqual(frame.shaderSha256, original.configuration.shaderSha256)
            XCTAssertEqual(frame.packedWeightHashes, original.configuration.weightHashes)
            let input = frame.input.contents().assumingMemoryBound(to: Float.self)
            for pixel in 0..<extent.pixels {
                for channel in 0..<3 { input[pixel * 4 + channel] = golden.input[channel * extent.pixels + pixel] }
                input[pixel * 4 + 3] = 0
            }
            let command = device.makeCommandQueue()!.makeCommandBuffer()!
            try frame.encode(command: command); command.commit(); command.waitUntilCompleted()
            XCTAssertEqual(command.status, .completed); XCTAssertNil(command.error)
            XCTAssertEqual(frame.validationFlagsAfterCompletion(), 0)
            let stages = try frame.diagnosticStagesAfterCompletion()
            XCTAssertTrue(try Validation.qualify(stages, expected: golden.expected, precision: precision, declaredPrecision: precision.rawValue).allSatisfy(\.passed))
            try original.loadInput(golden.input); let reference = try original.capture()
            for (actual, expected) in zip(stages, reference.checkpoints) {
                let comparison = try Validation.compare(actual: actual, expected: expected, tolerance: 0)
                XCTAssertTrue(comparison.passed, "\(precision) \(actual.name): \(comparison)")
            }
            input[0] = .nan
            let poisoned = device.makeCommandQueue()!.makeCommandBuffer()!
            try frame.encode(command: poisoned); poisoned.commit(); poisoned.waitUntilCompleted()
            XCTAssertEqual(poisoned.status, .completed)
            XCTAssertNotEqual(frame.validationFlagsAfterCompletion(), 0)
        }
    }
}