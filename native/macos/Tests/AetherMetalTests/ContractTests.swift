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
}