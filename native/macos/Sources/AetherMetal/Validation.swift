import Foundation

public struct StageTensor: Codable {
    public let name: String
    public let width: Int
    public let height: Int
    public let channels: Int
    public var layout: String = "CHW"
    public var values: [Float]
}

public struct StageComparison: Codable {
    public let stage: String
    public let elements: Int
    public let tolerance: Double
    public let maxAbsError: Double?
    public let meanAbsError: Double?
    public let failingElements: Int
    public let nonFinite: Int
    public let bitIdentical: Bool
    public let worstIndex: Int?
    public let edgeFailures: Int
    public let interiorFailures: Int
    public let passed: Bool
}

public enum Validation {
    public static func rgba(_ bytes: Data, expected: StageTensor, precision: Precision) throws -> StageComparison {
        let pixels = expected.width * expected.height
        try require(bytes.count == pixels * 4 && expected.channels == 3, "Wrong RGBA extent")
        let raw = [UInt8](bytes)
        try require((0..<pixels).allSatisfy { raw[$0 * 4 + 3] == 255 }, "Non-opaque RGBA output")
        let planar = (0..<3).flatMap { channel in (0..<pixels).map { Float(raw[$0 * 4 + channel]) / 255 } }
        let tensor = StageTensor(name: expected.name, width: expected.width, height: expected.height, channels: 3, values: planar)
        return try compare(actual: tensor, expected: expected, tolerance: max(precision.tolerance, 1.5 / 255))
    }

    public static func compare(actual: StageTensor, expected: StageTensor, tolerance: Double) throws -> StageComparison {
        try require(tolerance.isFinite && tolerance >= 0, "Invalid comparison tolerance")
        try require(actual.name == expected.name && actual.layout == "CHW" && expected.layout == "CHW", "Stage name/layout mismatch")
        try require(expected.width > 0 && expected.height > 0 && expected.width <= 2560 && expected.height <= 1440 &&
                    expected.channels > 0 && expected.channels <= 16, "Invalid expected extent")
        let elements = expected.width * expected.height * expected.channels
        try require(actual.width == expected.width && actual.height == expected.height && actual.channels == expected.channels &&
                    actual.values.count == elements && expected.values.count == elements, "Stage tensor shape mismatch")
        var maximum = 0.0, sum = 0.0, failures = 0, nonFinite = 0, worst: Int?, edges = 0
        var identical = true
        for index in 0..<elements {
            let observed = actual.values[index], reference = expected.values[index]
            identical = identical && observed.bitPattern == reference.bitPattern
            let difference = abs(Double(observed) - Double(reference))
            let invalid = !difference.isFinite || !observed.isFinite || !reference.isFinite
            if invalid { nonFinite += 1 }
            else {
                sum += difference
                if difference > maximum { maximum = difference; worst = index }
            }
            if invalid || difference > tolerance {
                failures += 1
                let pixel = index % (expected.width * expected.height), column = pixel % expected.width, row = pixel / expected.width
                if column < 2 || row < 2 || column >= expected.width - 2 || row >= expected.height - 2 { edges += 1 }
            }
        }
        return StageComparison(stage: actual.name, elements: elements, tolerance: tolerance,
            maxAbsError: nonFinite == 0 ? maximum : nil, meanAbsError: nonFinite == 0 ? sum / Double(elements) : nil,
            failingElements: failures, nonFinite: nonFinite, bitIdentical: identical, worstIndex: worst,
            edgeFailures: edges, interiorFailures: failures - edges, passed: failures == 0)
    }

    public static func qualify(_ actual: [StageTensor], expected: [StageTensor], precision: Precision,
                               declaredPrecision: String) throws -> [StageComparison] {
        let names = ["stem", "body.0", "body.1", "final"]
        try require(actual.map(\.name) == names && expected.map(\.name) == names, "Missing or reordered checkpoint")
        try require(declaredPrecision == precision.rawValue, "Precision metadata mismatch")
        return try zip(actual, expected).map { try compare(actual: $0, expected: $1, tolerance: precision.tolerance) }
    }
}