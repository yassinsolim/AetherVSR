import AetherMetal
import Foundation

struct Artifact: Codable { let file: String; let bytes: Int; let sha256: String }
struct Qualification: Encodable {
    let schema = "aethervsr.m13.phase15-golden/1"
    let configuration: OptimizedConfiguration
    let goldenBytesSha256 = Golden.fileHash
    let modelIdentity = ProductionModel.identity
    let inputFloat32Sha256: String
    let webgpuReferenceSha256: String
    let golden: [StageComparison]
    let webgpu: [StageComparison]
    let normalizedGolden: StageComparison
    let normalizedWebGPU: StageComparison
    let webgpuRGBAByteIdentical: Bool
    let fastRGBAByteIdentical: Bool
    let artifacts: [Artifact]
    let outcome: String
}
struct Performance: Encodable {
    let schema = "aethervsr.m13.phase15-timing/1"
    let configuration: OptimizedConfiguration
    let inputFloat32Sha256: String
    let qualificationSha256: String
    let startedAt: String
    let finishedAt: String
    let timing: OptimizedTiming
    let postRGBAHash: String
    let postRGBABytes: Int
    let postAlphaOpaque: Bool
    let postDiagnosticRGBAEqual: Bool
    let postStages: [OutputAudit]
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

var output: URL?
var began = false
var phase = "arguments"
var preflightThermalState: Int?
do {
    let arguments = Array(CommandLine.arguments.dropFirst())
    guard arguments.count == 5, ["qualify", "explore", "binding"].contains(arguments[0]),
          let candidate = MetalCandidate(rawValue: arguments[1]), let precision = Precision(rawValue: arguments[2]) else {
        throw QualificationError("Usage: aether-metal-opt qualify|explore|binding B|C|D|E|F|G f32|f16 NEW_OUTPUT WEBGPU_REFERENCE")
    }
    let directory = URL(fileURLWithPath: arguments[3], isDirectory: true)
    guard !FileManager.default.fileExists(atPath: directory.path) else { throw QualificationError("Attempt already exists") }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    output = directory
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    func write<Value: Encodable>(_ name: String, _ value: Value) throws -> Data {
        let data = try encoder.encode(value)
        try data.write(to: directory.appendingPathComponent(name), options: .withoutOverwriting)
        return data
    }
    func tensorBytes(_ values: [Float]) -> Data { values.map { $0.bitPattern.littleEndian }.withUnsafeBytes { Data($0) } }
    phase = "qualification"
    let model = try ProductionModel(data: Data(contentsOf: URL(fileURLWithPath: "public/models/aethersr-c16d2.json")))
    let golden = try Golden.load(Data(contentsOf: URL(fileURLWithPath: "public/models/golden-c16d2.json")))
    let referenceData = try Data(contentsOf: URL(fileURLWithPath: arguments[4]))
    let reference = try WebGPUReference.load(referenceData, golden: golden, precision: precision)
    let extent = try Extent(width: golden.width, height: golden.height)
    let engine = try OptimizedEngine(model: model, extent: extent, precision: precision, candidate: candidate, diagnostic: true)
    defer { engine.destroy() }
    try engine.loadInput(golden.input)
    let captured = try engine.capture()
    let fast = try OptimizedEngine(model: model, extent: extent, precision: precision, candidate: candidate, diagnostic: false)
    defer { fast.destroy() }
    try fast.loadInput(golden.input)
    let fastOutput = try fast.capture()
    let comparisons = try Validation.qualify(captured.checkpoints, expected: golden.expected, precision: precision, declaredPrecision: precision.rawValue)
    let parity = try Validation.qualify(captured.checkpoints, expected: reference.stages, precision: precision, declaredPrecision: reference.precision)
    let normalized = try Validation.rgba(captured.rgba, expected: golden.expected.last!, precision: precision)
    let referenceRGBA = try Validation.normalizedRGBA(Data(reference.rgba), width: golden.width * 2, height: golden.height * 2)
    let normalizedParity = try Validation.rgba(captured.rgba, expected: referenceRGBA, precision: precision)
    var artifacts: [Artifact] = []
    func artifact(_ name: String, _ data: Data) throws {
        try data.write(to: directory.appendingPathComponent(name), options: .withoutOverwriting)
        artifacts.append(Artifact(file: name, bytes: data.count, sha256: sha256(data)))
    }
    for stage in captured.checkpoints { try artifact(stage.name + ".f32le", tensorBytes(stage.values)) }
    try artifact("output.rgba8", captured.rgba); try artifact("fast.rgba8", fastOutput.rgba)
    let passed = comparisons.allSatisfy(\.passed) && parity.allSatisfy(\.passed) && normalized.passed && normalizedParity.passed && fastOutput.rgba == captured.rgba
    let qualification = Qualification(configuration: engine.configuration, inputFloat32Sha256: sha256(tensorBytes(golden.input)),
        webgpuReferenceSha256: sha256(referenceData), golden: comparisons, webgpu: parity, normalizedGolden: normalized,
        normalizedWebGPU: normalizedParity, webgpuRGBAByteIdentical: captured.rgba == Data(reference.rgba),
        fastRGBAByteIdentical: fastOutput.rgba == captured.rgba, artifacts: artifacts, outcome: passed ? "PASS" : "FAIL")
    let qualificationData = try write("qualification.json", qualification)
    guard passed else { throw QualificationError("Candidate numerical qualification failed") }
    if let mode = OptimizedMeasurement(rawValue: arguments[0]) {
        guard precision == .f16 else { throw QualificationError("Only f16 is a performance target") }
        engine.destroy(); fast.destroy()
        phase = "configuration"
        let largeExtent = try Extent(width: 1280, height: 720)
        let input = golden.tiledInput(extent: largeExtent)
        let inputHash = sha256(tensorBytes(input))
        guard inputHash == "e43b54d00e708a81b1e537e5345e72195300b5ab3dde75f2e05204d273953c19" else { throw QualificationError("Timing input identity mismatch") }
        let workload = try OptimizedEngine(model: model, extent: largeExtent, precision: .f16, candidate: candidate, diagnostic: false)
        defer { workload.destroy() }
        try workload.loadInput(input)
        let progressPath = directory.appendingPathComponent("progress.jsonl")
        try Data().write(to: progressPath, options: .withoutOverwriting)
        let progress = try FileHandle(forWritingTo: progressPath); defer { try? progress.close() }
        let compact = JSONEncoder(); compact.outputFormatting = [.sortedKeys]
        phase = "thermal-preflight"
        let startedAt = ISO8601DateFormatter().string(from: Date())
        let timing = try workload.measure(mode) { event in
            if event.phase == "preflight" { preflightThermalState = event.thermalState }
            if event.phase == "begin" { began = true; phase = "measurement" }
            try progress.write(contentsOf: compact.encode(event)); try progress.write(contentsOf: Data([10])); try progress.synchronize()
        }
        let finishedAt = ISO8601DateFormatter().string(from: Date())
        phase = "post-capture"
        let final = try workload.capture()
        let raw = [UInt8](final.rgba)
        let opaque = raw.count == largeExtent.pixels * 16 && (0..<(raw.count / 4)).allSatisfy { raw[$0 * 4 + 3] == 255 }
        workload.destroy()
        let auditEngine = try OptimizedEngine(model: model, extent: largeExtent, precision: .f16, candidate: candidate, diagnostic: true)
        defer { auditEngine.destroy() }
        try auditEngine.loadInput(input)
        let audit = try auditEngine.capture()
        let outputAudits = audit.checkpoints.map { stage in
            OutputAudit(stage: stage.name, elements: stage.values.count, nonFinite: stage.values.filter { !$0.isFinite }.count,
                outsideRange: stage.values.filter { $0 < (stage.name == "final" ? 0 : -1) || $0 > 1 }.count, sha256: sha256(tensorBytes(stage.values)))
        }
        let performance = Performance(configuration: workload.configuration, inputFloat32Sha256: inputHash,
            qualificationSha256: sha256(qualificationData), startedAt: startedAt, finishedAt: finishedAt, timing: timing,
            postRGBAHash: sha256(final.rgba), postRGBABytes: final.rgba.count, postAlphaOpaque: opaque,
            postDiagnosticRGBAEqual: audit.rgba == final.rgba, postStages: outputAudits,
            outcome: timing.wholeGraph.complete && opaque && audit.rgba == final.rgba && outputAudits.allSatisfy(\.passed) ? "PASS" : "FAIL")
        _ = try write("timing.json", performance)
        guard performance.outcome == "PASS" else { throw QualificationError("Incomplete timing/output audit") }
    }
    print("\(candidate.rawValue) \(precision.rawValue) \(arguments[0]): PASS")
} catch {
    if let output = output {
        let failure: [String: Any] = ["schema": "aethervsr.m13.phase15-failure/1", "phase": phase, "began": began,
            "thermalState": preflightThermalState ?? ProcessInfo.processInfo.thermalState.rawValue,
            "error": String(describing: error), "outcome": String(describing: error) == "THERMAL_NOT_STARTED" ? "NOT_STARTED" : "FAIL"]
        if let bytes = try? JSONSerialization.data(withJSONObject: failure, options: [.sortedKeys, .prettyPrinted]) {
            try? bytes.write(to: output.appendingPathComponent("failure.json"), options: .withoutOverwriting)
        }
    }
    FileHandle.standardError.write(Data("\(error)\n".utf8)); exit(1)
}