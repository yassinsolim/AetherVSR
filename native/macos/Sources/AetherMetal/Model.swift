import CryptoKit
import Foundation

public struct QualificationError: Error, CustomStringConvertible {
    public let description: String
    public init(_ description: String) { self.description = description }
}

func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw QualificationError(message) }
}

public func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

public enum Precision: String, Codable, CaseIterable {
    case f32, f16
    public var tolerance: Double { self == .f32 ? 1e-3 : 5e-2 }
    public var bytesPerElement: Int { self == .f32 ? 4 : 2 }
}

struct Layer: Decodable, Equatable {
    let name: String
    let type: String
    let kernel: Int?
    let input: Int?
    let output: Int?
    let padding: Int?
    let activation: String?
    let scale: Int?
    let clamp: [Float]?
    let residual: String?

    enum CodingKeys: String, CodingKey {
        case name, type, kernel, padding, activation, scale, clamp, residual
        case input = "in", output = "out"
    }
}

struct ModelDocument: Decodable {
    struct Normalisation: Decodable { let mean: [Float]; let scale: [Float]; let range: String }
    let architecture: String
    let architectureVersion: Int
    let scale: Int
    let inChannels: Int
    let outChannels: Int
    let features: Int
    let depth: Int
    let parameters: Int
    let sha256: String
    let layers: [Layer]
    let normalisation: Normalisation
    let weights: [String: [Float]]
}

public struct ProductionModel {
    public static let fileHash = "d76fae7a295cdcdaecb44e39f8c87ff68a59ca1e07fc7cfc347d252d3cad358a"
    public static let identity = "9154d9490b02d8fdf136edd990217cfb6f0e9956ed928da7179d7781832c8e02"
    public static let tensorCounts = ["stem.weight": 1200, "stem.bias": 16,
        "body.0.weight": 2304, "body.0.bias": 16, "body.1.weight": 2304,
        "body.1.bias": 16, "head.weight": 432, "head.bias": 3]
    let weights: [String: [Float]]

    public init(data: Data) throws {
        try require(sha256(data) == Self.fileHash, "Production model byte SHA mismatch")
        try self.init(document: JSONDecoder().decode(ModelDocument.self, from: data))
    }

    init(document: ModelDocument) throws {
        try require(document.architecture == "aethersr-resizeconv" && document.architectureVersion == 2,
                    "Wrong model architecture")
        try require(document.scale == 2 && document.inChannels == 3 && document.outChannels == 3 &&
                    document.features == 16 && document.depth == 2 && document.parameters == 6291,
                    "Wrong C16D2 dimensions")
        try require(document.sha256 == Self.identity, "Wrong embedded model identity")
        try require(document.normalisation.mean == [0, 0, 0] && document.normalisation.scale == [1, 1, 1] &&
                    document.normalisation.range == "[0,1] RGB", "Wrong input normalization")
        try require(document.layers.map(\.name) == ["stem", "body.0", "body.1", "upsample", "head"],
                    "Wrong model stage order")
        for (index, layer) in document.layers.enumerated() {
            if index == 3 {
                try require(layer.type == "nearest" && layer.scale == 2, "Wrong reconstruction scale")
            } else {
                try require(layer.type == "conv" && layer.kernel == (index == 0 ? 5 : 3) &&
                            layer.padding == (index == 0 ? 2 : 1) && layer.input == (index == 0 ? 3 : 16) &&
                            layer.output == (index == 4 ? 3 : 16) && layer.activation == (index == 4 ? "none" : "tanh"),
                            "Wrong convolution semantics: \(layer.name)")
                if index == 4 {
                    try require(layer.clamp == [0, 1] && layer.residual == "nearest-upsampled input added before clamp",
                                "Wrong head residual/clamp")
                }
            }
        }
        try require(Set(document.weights.keys) == Set(Self.tensorCounts.keys), "Wrong model tensor names")
        for (name, count) in Self.tensorCounts {
            let values = document.weights[name]!
            try require(values.count == count && values.allSatisfy(\.isFinite), "Invalid weight tensor: \(name)")
        }
        weights = document.weights
    }
}

public struct Extent: Equatable, Encodable {
    public let width: Int
    public let height: Int
    public var pixels: Int { width * height }
    public init(width: Int, height: Int) throws {
        try require(width > 0 && height > 0 && width <= 1280 && height <= 720, "Extent outside Phase-1 bounds")
        self.width = width; self.height = height
    }
}

public struct Golden: Decodable {
    public static let fileHash = "7ffe8d5c26ef02f605c04e9057dd5ef9767dc83e84695211eff39f04a1e3a759"
    public let modelSha256: String
    public let features: Int
    public let depth: Int
    public let width: Int
    public let height: Int
    public let input: [Float]
    public let stages: [String: [Float]]
    public let output: [Float]

    public static func load(_ data: Data) throws -> Golden {
        try require(sha256(data) == fileHash, "Trusted golden byte SHA mismatch")
        let golden = try JSONDecoder().decode(Golden.self, from: data)
        try golden.validate()
        return golden
    }

    func validate() throws {
        let extent = try Extent(width: width, height: height)
        try require(modelSha256 == ProductionModel.identity && features == 16 && depth == 2,
                    "Golden model identity/architecture mismatch")
        try require(input.count == extent.pixels * 3 && output.count == extent.pixels * 12,
                    "Wrong golden tensor shape")
        try require(Set(stages.keys) == Set(["stem", "body.0", "body.1"]), "Missing/unknown golden checkpoint")
        for values in stages.values { try require(values.count == extent.pixels * 16, "Wrong golden stage shape") }
        try require((input + output + stages.values.flatMap { $0 }).allSatisfy(\.isFinite), "Nonfinite golden value")
    }

    public var expected: [StageTensor] {
        ["stem", "body.0", "body.1"].map {
            StageTensor(name: $0, width: width, height: height, channels: 16, values: stages[$0]!)
        } + [StageTensor(name: "final", width: width * 2, height: height * 2, channels: 3,
                         values: Self.planar(output, width: width * 2, height: height * 2, channels: 3))]
    }

    public func tiledInput(extent: Extent) -> [Float] {
        var result = [Float](repeating: 0, count: extent.pixels * 3)
        for channel in 0..<3 {
            let sourcePlane = channel * width * height
            let destinationPlane = channel * extent.pixels
            for row in 0..<extent.height {
                let sourceRow = sourcePlane + (row % height) * width
                let destinationRow = destinationPlane + row * extent.width
                for column in 0..<extent.width {
                    result[destinationRow + column] = input[sourceRow + column % width]
                }
            }
        }
        return result
    }

    public static func planar(_ interleaved: [Float], width: Int, height: Int, channels: Int) -> [Float] {
        let pixels = width * height
        return (0..<channels).flatMap { channel in (0..<pixels).map { interleaved[$0 * channels + channel] } }
    }
}