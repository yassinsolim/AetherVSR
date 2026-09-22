import AetherMetal
import Foundation

struct Artifact: Encodable { let file: String; let bytes: Int; let sha256: String }
struct BackendComparison: Encodable {
    let referenceSha256: String
    let stages: [StageComparison]
    let normalizedRGBA: StageComparison
    let rgbaByteIdentical: Bool
}
struct Evidence: Encodable {
    let schema = "aethervsr.m13.metal-golden/1"
    let modelBytesSha256 = ProductionModel.fileHash
    let modelIdentity = ProductionModel.identity
    let goldenBytesSha256 = Golden.fileHash
    let inputFloat32Sha256: String
    let precision: String
    let device: String
    let unifiedMemory: Bool
    let stages: [StageComparison]
    let normalizedRGBA: StageComparison
    let webgpu: BackendComparison?
    let artifacts: [Artifact]
    let outcome: String
}

do {
    let arguments = Array(CommandLine.arguments.dropFirst())
        guard (arguments.count == 8 || arguments.count == 10) && arguments[0] == "--model" && arguments[2] == "--input" && arguments[4] == "--precision" && arguments[6] == "--output",
                    (arguments.count == 8 || arguments[8] == "--webgpu"),
          let precision = Precision(rawValue: arguments[5]) else {
                throw QualificationError("Usage: aether-metal --model MODEL.json --input GOLDEN.json --precision f32|f16 --output NEW_DIRECTORY [--webgpu REFERENCE.json]")
    }
    let model = try ProductionModel(data: Data(contentsOf: URL(fileURLWithPath: arguments[1])))
    let golden = try Golden.load(Data(contentsOf: URL(fileURLWithPath: arguments[3])))
    let referenceBytes = arguments.count == 10 ? try Data(contentsOf: URL(fileURLWithPath: arguments[9])) : nil
    let reference = try referenceBytes.map { try WebGPUReference.load($0, golden: golden, precision: precision) }
    let output = URL(fileURLWithPath: arguments[7], isDirectory: true)
    guard !FileManager.default.fileExists(atPath: output.path) else { throw QualificationError("Evidence output already exists") }
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let engine = try MetalEngine(model: model, extent: Extent(width: golden.width, height: golden.height), precision: precision)
    defer { engine.destroy() }
    try engine.loadInput(golden.input)
    let capture = try engine.capture()
    let comparisons = try Validation.qualify(capture.checkpoints, expected: golden.expected, precision: precision, declaredPrecision: precision.rawValue)
    let normalized = try Validation.rgba(capture.rgba, expected: golden.expected.last!, precision: precision)
    let backend: BackendComparison?
    if let reference = reference, let bytes = referenceBytes {
        let stages = try Validation.qualify(capture.checkpoints, expected: reference.stages, precision: precision, declaredPrecision: reference.precision)
        let rgba = try Validation.normalizedRGBA(Data(reference.rgba), width: golden.width * 2, height: golden.height * 2)
        backend = BackendComparison(referenceSha256: sha256(bytes), stages: stages,
            normalizedRGBA: try Validation.rgba(capture.rgba, expected: rgba, precision: precision),
            rgbaByteIdentical: capture.rgba == Data(reference.rgba))
    } else { backend = nil }
    var artifacts: [Artifact] = []
    func write(_ name: String, _ bytes: Data) throws {
        try bytes.write(to: output.appendingPathComponent(name), options: .withoutOverwriting)
        artifacts.append(Artifact(file: name, bytes: bytes.count, sha256: sha256(bytes)))
    }
    for stage in capture.checkpoints {
        try write(stage.name + ".f32le", stage.values.map { $0.bitPattern.littleEndian }.withUnsafeBytes { Data($0) })
    }
    try write("output.rgba8", capture.rgba)
    let inputHash = sha256(golden.input.map { $0.bitPattern.littleEndian }.withUnsafeBytes { Data($0) })
    let passed = comparisons.allSatisfy(\.passed) && normalized.passed &&
        (backend.map { $0.stages.allSatisfy(\.passed) && $0.normalizedRGBA.passed } ?? true)
    let evidence = Evidence(inputFloat32Sha256: inputHash, precision: precision.rawValue, device: engine.device.name, unifiedMemory: engine.device.hasUnifiedMemory,
        stages: comparisons, normalizedRGBA: normalized, webgpu: backend, artifacts: artifacts, outcome: passed ? "PASS" : "FAIL")
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let bytes = try encoder.encode(evidence)
    try bytes.write(to: output.appendingPathComponent("result.json"), options: .withoutOverwriting)
    print(String(decoding: bytes, as: UTF8.self))
    if evidence.outcome != "PASS" { exit(1) }
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}