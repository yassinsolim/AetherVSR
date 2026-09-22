import AetherMetal
import Foundation

struct Artifact: Encodable { let file: String; let bytes: Int; let sha256: String }
struct Evidence: Encodable {
    let schema = "aethervsr.m13.metal-golden/1"
    let modelBytesSha256 = ProductionModel.fileHash
    let modelIdentity = ProductionModel.identity
    let goldenBytesSha256 = Golden.fileHash
    let precision: String
    let device: String
    let unifiedMemory: Bool
    let stages: [StageComparison]
    let normalizedRGBA: StageComparison
    let artifacts: [Artifact]
    let outcome: String
}

do {
    let arguments = Array(CommandLine.arguments.dropFirst())
    guard arguments.count == 8 && arguments[0] == "--model" && arguments[2] == "--input" && arguments[4] == "--precision" && arguments[6] == "--output",
          let precision = Precision(rawValue: arguments[5]) else {
        throw QualificationError("Usage: aether-metal --model MODEL.json --input GOLDEN.json --precision f32 --output NEW_DIRECTORY")
    }
    let model = try ProductionModel(data: Data(contentsOf: URL(fileURLWithPath: arguments[1])))
    let golden = try Golden.load(Data(contentsOf: URL(fileURLWithPath: arguments[3])))
    let output = URL(fileURLWithPath: arguments[7], isDirectory: true)
    guard !FileManager.default.fileExists(atPath: output.path) else { throw QualificationError("Evidence output already exists") }
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let engine = try MetalEngine(model: model, extent: Extent(width: golden.width, height: golden.height), precision: precision)
    defer { engine.destroy() }
    try engine.loadInput(golden.input)
    let capture = try engine.capture()
    let comparisons = try Validation.qualify(capture.checkpoints, expected: golden.expected, precision: precision, declaredPrecision: precision.rawValue)
    let normalized = try Validation.rgba(capture.rgba, expected: golden.expected.last!, precision: precision)
    var artifacts: [Artifact] = []
    func write(_ name: String, _ bytes: Data) throws {
        try bytes.write(to: output.appendingPathComponent(name), options: .withoutOverwriting)
        artifacts.append(Artifact(file: name, bytes: bytes.count, sha256: sha256(bytes)))
    }
    for stage in capture.checkpoints {
        try write(stage.name + ".f32le", stage.values.withUnsafeBytes { Data($0) })
    }
    try write("output.rgba8", capture.rgba)
    let evidence = Evidence(precision: precision.rawValue, device: engine.device.name, unifiedMemory: engine.device.hasUnifiedMemory,
        stages: comparisons, normalizedRGBA: normalized, artifacts: artifacts, outcome: comparisons.allSatisfy(\.passed) && normalized.passed ? "PASS" : "FAIL")
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let bytes = try encoder.encode(evidence)
    try bytes.write(to: output.appendingPathComponent("result.json"), options: .withoutOverwriting)
    print(String(decoding: bytes, as: UTF8.self))
    if evidence.outcome != "PASS" { exit(1) }
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}