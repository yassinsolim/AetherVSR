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

struct OutputAudit: Encodable {
    let stage: String
    let elements: Int
    let nonFinite: Int
    let outsideRange: Int
    let sha256: String
    var passed: Bool { nonFinite == 0 && outsideRange == 0 }
}

struct PerformanceEvidence: Encodable {
    let schema = "aethervsr.m13.metal-timing/1"
    let modelBytesSha256 = ProductionModel.fileHash
    let modelIdentity = ProductionModel.identity
    let goldenBytesSha256 = Golden.fileHash
    let precision: String
    let device: String
    let inputExtent: Extent
    let inputFloat32Sha256: String
    let correctnessSha256: String
    let thermalStateBefore: Int
    let thermalStateAfter: Int
    let startedAt: String
    let finishedAt: String
    let timings: TimingCapture
    let postTimingStages: [OutputAudit]
    let postTimingRGBA: StageComparison
    let postTimingRGBAHash: String
    let outcome: String
}

do {
    var arguments = Array(CommandLine.arguments.dropFirst())
    let benchmark = arguments.last == "--benchmark"
    if benchmark { arguments.removeLast() }
        guard (arguments.count == 8 || arguments.count == 10) && arguments[0] == "--model" && arguments[2] == "--input" && arguments[4] == "--precision" && arguments[6] == "--output",
                    (arguments.count == 8 || arguments[8] == "--webgpu"),
          let precision = Precision(rawValue: arguments[5]) else {
                throw QualificationError("Usage: aether-metal --model MODEL.json --input GOLDEN.json --precision f32|f16 --output NEW_DIRECTORY [--webgpu REFERENCE.json] [--benchmark]")
    }
            if benchmark && arguments.count != 10 { throw QualificationError("Benchmark requires a verified WebGPU reference") }
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
    if benchmark {
        engine.destroy()
        let extent = try Extent(width: 1280, height: 720)
        let input = golden.tiledInput(extent: extent)
        let inputHash = sha256(input.map { $0.bitPattern.littleEndian }.withUnsafeBytes { Data($0) })
        let workload = try MetalEngine(model: model, extent: extent, precision: precision)
        defer { workload.destroy() }
        try workload.loadInput(input)
        let startedAt = ISO8601DateFormatter().string(from: Date())
        let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
        let timings = try workload.benchmark()
        let thermalAfter = ProcessInfo.processInfo.thermalState.rawValue
        let finishedAt = ISO8601DateFormatter().string(from: Date())
        let actual = try workload.capture()
        let audits = actual.checkpoints.map { stage in
            let lower: Float = stage.name == "final" ? 0 : -1
            return OutputAudit(stage: stage.name, elements: stage.values.count,
                nonFinite: stage.values.filter { !$0.isFinite }.count,
                outsideRange: stage.values.filter { $0 < lower || $0 > 1 }.count,
                sha256: sha256(stage.values.map { $0.bitPattern.littleEndian }.withUnsafeBytes { Data($0) }))
        }
        let rgba = try Validation.rgba(actual.rgba, expected: actual.checkpoints.last!, precision: .f32)
        let performance = PerformanceEvidence(precision: precision.rawValue, device: workload.device.name,
            inputExtent: extent, inputFloat32Sha256: inputHash, correctnessSha256: sha256(bytes),
            thermalStateBefore: thermalBefore, thermalStateAfter: thermalAfter, startedAt: startedAt, finishedAt: finishedAt,
            timings: timings, postTimingStages: audits, postTimingRGBA: rgba, postTimingRGBAHash: sha256(actual.rgba),
            outcome: timings.complete && audits.allSatisfy(\.passed) && rgba.passed ? "PASS" : "FAIL")
        let timingBytes = try encoder.encode(performance)
        try timingBytes.write(to: output.appendingPathComponent("timing.json"), options: .withoutOverwriting)
        print("Metal timing \(precision.rawValue): \(performance.outcome)")
        if performance.outcome != "PASS" { exit(1) }
    }
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}