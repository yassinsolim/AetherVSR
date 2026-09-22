import Foundation
import XCTest
@testable import AetherMetal

final class ContractTests: XCTestCase {
    private var root: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }
    private func data(_ name: String) throws -> Data { try Data(contentsOf: root.appendingPathComponent("public/models/\(name)")) }

    func testProductionIdentityAndShapes() throws {
        let bytes = try data("aethersr-c16d2.json")
        let model = try ProductionModel(data: bytes)
        XCTAssertEqual(model.weights.values.reduce(0) { $0 + $1.count }, 6291)
        let golden = try Golden.load(data("golden-c16d2.json"))
        XCTAssertEqual(golden.expected.map(\.name), ["stem", "body.0", "body.1", "final"])
        XCTAssertEqual(golden.expected[0].values.count, 6144)
        XCTAssertEqual(golden.expected[3].values.count, 4608)
        let extent = try Extent(width: golden.width * 2 + 1, height: golden.height * 2 + 1)
        let tiled = golden.tiledInput(extent: extent)
        XCTAssertEqual(tiled.count, extent.pixels * 3)
        for channel in 0..<3 { for row in 0..<extent.height { for column in 0..<extent.width {
            XCTAssertEqual(tiled[channel * extent.pixels + row * extent.width + column],
                golden.input[channel * golden.width * golden.height + (row % golden.height) * golden.width + column % golden.width])
        } } }
        let fullSize = golden.tiledInput(extent: try Extent(width: 1280, height: 720))
        let fullSizeBytes = fullSize.map { $0.bitPattern.littleEndian }.withUnsafeBytes { Data($0) }
        XCTAssertEqual(sha256(fullSizeBytes), "e43b54d00e708a81b1e537e5345e72195300b5ab3dde75f2e05204d273953c19")
    }

    func testWrongHashAndCorruptWeightRejected() throws {
        var bytes = try data("aethersr-c16d2.json")
        bytes.append(32)
        XCTAssertThrowsError(try ProductionModel(data: bytes))
        var document = try JSONSerialization.jsonObject(with: data("aethersr-c16d2.json")) as! [String: Any]
        var weights = document["weights"] as! [String: [Double]]
        weights["stem.weight"]![0] += 1
        document["weights"] = weights
        XCTAssertThrowsError(try ProductionModel(data: JSONSerialization.data(withJSONObject: document)))
    }

    func testMalformedModelAndGoldenSchemasRejected() throws {
        let bytes = try data("aethersr-c16d2.json")
        for field in ["features", "depth", "inChannels", "outChannels", "scale", "parameters"] {
            var object = try JSONSerialization.jsonObject(with: bytes) as! [String: Any]
            object[field] = 99
            let decoded = try JSONDecoder().decode(ModelDocument.self, from: JSONSerialization.data(withJSONObject: object))
            XCTAssertThrowsError(try ProductionModel(document: decoded))
        }
        var object = try JSONSerialization.jsonObject(with: data("golden-c16d2.json")) as! [String: Any]
        object["width"] = 23
        XCTAssertThrowsError(try JSONDecoder().decode(Golden.self, from: JSONSerialization.data(withJSONObject: object)).validate())
        object = try JSONSerialization.jsonObject(with: data("golden-c16d2.json")) as! [String: Any]
        var stages = object["stages"] as! [String: Any]; stages.removeValue(forKey: "stem"); object["stages"] = stages
        XCTAssertThrowsError(try JSONDecoder().decode(Golden.self, from: JSONSerialization.data(withJSONObject: object)).validate()) {
            XCTAssertEqual(String(describing: $0), "Missing/unknown golden checkpoint")
        }
    }

    func testFalsificationAndPrecisionMetadata() throws {
        let expected = try Golden.load(data("golden-c16d2.json")).expected
        for precision in Precision.allCases {
            XCTAssertTrue(try Validation.qualify(expected, expected: expected, precision: precision, declaredPrecision: precision.rawValue).allSatisfy(\.passed))
            for value in [Float.nan, Float.infinity, -Float.infinity, Float(10)] {
                var wrong = expected; wrong[0].values[0] = value
                XCTAssertFalse(try Validation.qualify(wrong, expected: expected, precision: precision, declaredPrecision: precision.rawValue).allSatisfy(\.passed))
            }
            var perturbed = expected; perturbed[1].values[7] += 1
            XCTAssertFalse(try Validation.qualify(expected, expected: perturbed, precision: precision, declaredPrecision: precision.rawValue).allSatisfy(\.passed))
        }
        XCTAssertThrowsError(try Validation.qualify(Array(expected.reversed()), expected: expected, precision: .f32, declaredPrecision: "f32"))
        XCTAssertThrowsError(try Validation.qualify(Array(expected.dropLast()), expected: expected, precision: .f32, declaredPrecision: "f32"))
        XCTAssertThrowsError(try Validation.qualify(expected, expected: expected, precision: .f32, declaredPrecision: "f16"))
        var wrong = expected; wrong[0].layout = "HWC"
        XCTAssertThrowsError(try Validation.qualify(wrong, expected: expected, precision: .f32, declaredPrecision: "f32"))
        wrong = expected; wrong[0].values.removeLast()
        XCTAssertThrowsError(try Validation.qualify(wrong, expected: expected, precision: .f32, declaredPrecision: "f32"))
    }

    func testExplicitPlanarLayoutAndBounds() throws {
        XCTAssertEqual(Golden.planar([1, 10, 100, 2, 20, 200, 3, 30, 300, 4, 40, 400], width: 2, height: 2, channels: 3),
                       [1, 2, 3, 4, 10, 20, 30, 40, 100, 200, 300, 400])
        XCTAssertThrowsError(try Extent(width: Int.max, height: 1))
        XCTAssertThrowsError(try Extent(width: 0, height: 2))
        XCTAssertThrowsError(try Extent(width: 1280, height: 721))
        XCTAssertEqual(try Extent(width: 1280, height: 720).pixels, 921600)
    }

    func testWebGPUReferenceIdentityAndFalsification() throws {
        let golden = try Golden.load(data("golden-c16d2.json"))
        let stages = try JSONSerialization.jsonObject(with: JSONEncoder().encode(golden.expected))
        var rgba: [UInt8] = []
        for pixel in 0..<(golden.width * golden.height * 4) {
            for channel in 0..<3 { rgba.append(UInt8((golden.output[pixel * 3 + channel] * 255).rounded())) }
            rgba.append(255)
        }
        let inputHash = sha256(golden.input.map { $0.bitPattern.littleEndian }.withUnsafeBytes { Data($0) })
        for precision in Precision.allCases {
            let valid: [String: Any] = ["schema": "aethervsr.m13.webgpu-golden/1", "outcome": "PASS",
                "modelBytesSha256": ProductionModel.fileHash, "modelIdentity": ProductionModel.identity,
                "goldenBytesSha256": Golden.fileHash, "inputFloat32Sha256": inputHash,
                "precision": precision.rawValue, "stages": stages, "rgba": rgba]
            func load(_ object: [String: Any]) throws -> WebGPUReference {
                try WebGPUReference.load(JSONSerialization.data(withJSONObject: object), golden: golden, precision: precision)
            }
            XCTAssertEqual(try load(valid).precision, precision.rawValue)
            for key in ["schema", "outcome", "modelBytesSha256", "modelIdentity", "goldenBytesSha256", "inputFloat32Sha256", "precision"] {
                var invalid = valid; invalid[key] = "wrong"
                XCTAssertThrowsError(try load(invalid))
            }
            var invalid = valid; var changed = golden.expected; changed[2].values[7] += 1
            invalid["stages"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(changed))
            XCTAssertThrowsError(try load(invalid))
            invalid = valid; var alpha = rgba; alpha[3] = 0; invalid["rgba"] = alpha
            XCTAssertThrowsError(try load(invalid))
        }
    }

    func testExecutionFailureIsTerminal() throws {
        var state = ExecutionState()
        try state.requireLive()
        try state.complete(success: true, message: "unused")
        XCTAssertThrowsError(try state.complete(success: false, message: "injected completion failure"))
        XCTAssertThrowsError(try state.requireLive())
        XCTAssertThrowsError(try state.complete(success: true, message: "must not revive"))
        state.destroy(); state.destroy()
        XCTAssertThrowsError(try state.requireLive())
    }

    func testGPUTimeAvailabilityAndPercentiles() throws {
        let samples = [1.0, 2.0, 3.0, 4.0].map { GPUInterval(start: 1, end: 1 + $0 / 1000) }
        let series = TimingSeries(name: "synthetic", samples: samples, windowMS: 20)
        XCTAssertTrue(series.complete); XCTAssertEqual(series.measuredSamples, 4)
        XCTAssertEqual(series.statisticsMS["p50"]!!, 2.5, accuracy: 1e-9)
        XCTAssertEqual(series.statisticsMS["p95"]!!, 3.85, accuracy: 1e-9)
        XCTAssertEqual(series.statisticsMS["max"]!!, 4, accuracy: 1e-9)
        for (start, end) in [(0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (.nan, 2.0), (1.0, .infinity)] {
            let invalid = GPUInterval(start: start, end: end)
            XCTAssertNil(invalid.milliseconds)
            let missing = TimingSeries(name: "synthetic", samples: [invalid], windowMS: 20)
            XCTAssertFalse(missing.complete); XCTAssertEqual(missing.status, "not measured")
            let json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(missing)) as! [String: Any]
            let statistics = json["statisticsMS"] as! [String: Any]
            XCTAssertTrue(statistics.values.allSatisfy { $0 is NSNull })
            let raw = (json["samples"] as! [[String: Any]])[0]
            XCTAssertTrue(raw.values.allSatisfy { $0 is NSNull })
        }
        let partial = TimingSeries(name: "synthetic", samples: samples + [GPUInterval(start: 0, end: 0)], windowMS: 20)
        XCTAssertFalse(partial.complete); XCTAssertEqual(partial.status, "partially measured")
    }

    func testConvolutionMetadataAndRGBARejections() throws {
        let bytes = try data("aethersr-c16d2.json")
        for (index, key, value) in [(0, "kernel", 3 as Any), (1, "padding", 0 as Any),
                                    (2, "activation", "relu" as Any), (3, "scale", 3 as Any),
                                    (4, "residual", "none" as Any), (4, "clamp", [-1, 1] as Any)] {
            var object = try JSONSerialization.jsonObject(with: bytes) as! [String: Any]
            var layers = object["layers"] as! [[String: Any]]
            layers[index][key] = value; object["layers"] = layers
            XCTAssertThrowsError(try ProductionModel(document: JSONDecoder().decode(ModelDocument.self,
                from: JSONSerialization.data(withJSONObject: object))))
        }
        let expected = StageTensor(name: "final", width: 1, height: 1, channels: 3, values: [0, 0.5, 1])
        XCTAssertTrue(try Validation.rgba(Data([0, 128, 255, 255]), expected: expected, precision: .f32).passed)
        XCTAssertThrowsError(try Validation.rgba(Data([0, 128, 255]), expected: expected, precision: .f32))
        XCTAssertThrowsError(try Validation.rgba(Data([0, 128, 255, 0]), expected: expected, precision: .f32))
        let oversized = StageTensor(name: "final", width: Int.max, height: 2, channels: 3, values: [])
        XCTAssertThrowsError(try Validation.rgba(Data(), expected: oversized, precision: .f32))
        XCTAssertEqual(Float16(1 + Float(1) / 2048).bitPattern, Float16(1).bitPattern)
        XCTAssertEqual(Float16(1 + Float(3) / 2048).bitPattern, Float16(1 + Float(2) / 1024).bitPattern)
    }
}