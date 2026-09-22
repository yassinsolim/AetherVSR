import Foundation

public struct KernelGeometry: Encodable, Equatable {
    public let groupX: Int
    public let groupY: Int
    public let blockX: Int
    public let blockY: Int
    public let outputBlock: Int
}

public enum MetalCandidate: String, Codable, CaseIterable {
    case b = "B", c = "C", d = "D", e = "E", f = "F", g = "G"

    public var tiled: Bool { self != .b }
    public var fusedActivation: Bool { self != .b && self != .c }
    public var fusedHead: Bool { self == .f || self == .g }
    public var spatial: Bool { self == .e || self == .f || self == .g }
    public var body: KernelGeometry {
        KernelGeometry(groupX: 8, groupY: self == .g ? 8 : 4, blockX: spatial ? 2 : 1,
                       blockY: spatial ? 2 : 1, outputBlock: fusedActivation ? 16 : 4)
    }
    public var stem: KernelGeometry {
        KernelGeometry(groupX: 8, groupY: spatial ? 8 : 4, blockX: spatial ? 2 : 1,
                       blockY: spatial ? 2 : 1, outputBlock: 16)
    }
    public var head: KernelGeometry {
        KernelGeometry(groupX: 8, groupY: 4, blockX: spatial ? 2 : 1,
                       blockY: fusedHead ? 4 : spatial ? 2 : 1, outputBlock: 3)
    }
}

enum PackedLayout {
    static func weights(_ source: [Float], inputChannels: Int, outputChannels: Int, kernel: Int) throws -> [Float] {
        try require((1...16).contains(inputChannels) && (1...16).contains(outputChannels) && [1, 3, 5].contains(kernel),
                    "Invalid packed convolution shape")
        let taps = kernel * kernel, groups = (inputChannels + 3) / 4
        try require(source.count == inputChannels * outputChannels * taps && source.allSatisfy(\.isFinite), "Invalid source weights")
        var packed = [Float](repeating: 0, count: groups * taps * outputChannels * 4)
        for group in 0..<groups { for tap in 0..<taps { for output in 0..<outputChannels { for lane in 0..<4 {
            let channel = group * 4 + lane
            if channel < inputChannels {
                packed[((group * taps + tap) * outputChannels + output) * 4 + lane] = source[(output * inputChannels + channel) * taps + tap]
            }
        } } } }
        return packed
    }

    static func pack(_ planar: [Float], extent: Extent, channels: Int) throws -> [Float] {
        try require((1...16).contains(channels) && planar.count == extent.pixels * channels && planar.allSatisfy(\.isFinite), "Invalid planar input")
        var packed = [Float](repeating: 0, count: ((channels + 3) / 4) * extent.pixels * 4)
        for channel in 0..<channels { for pixel in 0..<extent.pixels {
            packed[((channel / 4) * extent.pixels + pixel) * 4 + channel % 4] = planar[channel * extent.pixels + pixel]
        } }
        return packed
    }

    static func unpack(_ packed: [Float], extent: Extent, channels: Int) throws -> [Float] {
        try require((1...16).contains(channels) && packed.count == ((channels + 3) / 4) * extent.pixels * 4, "Invalid packed activation shape")
        var planar = [Float](repeating: 0, count: channels * extent.pixels)
        for channel in 0..<channels { for pixel in 0..<extent.pixels {
            planar[channel * extent.pixels + pixel] = packed[((channel / 4) * extent.pixels + pixel) * 4 + channel % 4]
        } }
        return planar
    }

    static func bytes(_ values: [Float], precision: Precision) throws -> Data {
        try require(values.allSatisfy(\.isFinite), "Nonfinite packed source")
        if precision == .f16 {
            let half = values.map { Float16($0) }
            try require(half.allSatisfy(\.isFinite), "Packed half overflow")
            return half.map { $0.bitPattern.littleEndian }.withUnsafeBytes { Data($0) }
        }
        return values.map { $0.bitPattern.littleEndian }.withUnsafeBytes { Data($0) }
    }
}

struct PackedParameters {
    let buffers: [String: Data]
    var hashes: [String: String] { buffers.mapValues { sha256($0) } }

    init(model: ProductionModel, precision: Precision) throws {
        var result: [String: Data] = [:]
        for layer in ["stem", "body.0", "body.1", "head"] {
            let inputChannels = layer == "stem" ? 3 : 16
            let outputChannels = layer == "head" ? 3 : 16
            let packed = try PackedLayout.weights(model.weights[layer + ".weight"]!, inputChannels: inputChannels,
                                                   outputChannels: outputChannels, kernel: layer == "stem" ? 5 : 3)
            result[layer + ".weight"] = try PackedLayout.bytes(packed, precision: precision)
            result[layer + ".bias"] = try PackedLayout.bytes(model.weights[layer + ".bias"]!, precision: precision)
        }
        buffers = result
    }

    func verify(_ actual: [String: Data], modelHash: String) throws {
        try require(modelHash == ProductionModel.fileHash && actual == buffers, "Packed parameter identity/corruption failure")
    }
}