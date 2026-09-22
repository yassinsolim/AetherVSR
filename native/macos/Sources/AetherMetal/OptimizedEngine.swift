import Foundation
import Metal

public struct OptimizedConfiguration: Encodable {
    public let candidate: MetalCandidate
    public let precision: Precision
    public let diagnostic: Bool
    public let extent: Extent
    public let device: String
    public let modelBytesSha256 = ProductionModel.fileHash
    public let shaderSha256: String
    public let weightHashes: [String: String]
    public let stem: KernelGeometry
    public let body: KernelGeometry
    public let head: KernelGeometry
    public let pipelineLimits: [String: [String: Int]]
    public let configuredBufferBytes: Int
    public let outputTextureBytes: Int
}

public enum OptimizedMeasurement: String, Codable {
    case explore, binding
    public var warmups: Int { self == .explore ? 5 : 10 }
    public var samples: Int { self == .explore ? 20 : 60 }
}

public struct MeasurementEvent: Encodable {
    public let phase: String
    public let index: Int
    public let thermalState: Int
    public let interval: GPUInterval?
}

public struct OptimizedTiming: Encodable {
    public let mode: OptimizedMeasurement
    public let warmups: Int
    public let requestedSamples: Int
    public let thermalStateBefore: Int
    public let thermalStateAfter: Int
    public let wholeGraph: TimingSeries
}

public final class OptimizedEngine {
    public let candidate: MetalCandidate
    public let precision: Precision
    public let extent: Extent
    public let diagnostic: Bool
    public let device: MTLDevice
    public private(set) var configuration: OptimizedConfiguration!
    private let queue: MTLCommandQueue
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private var buffers: [String: MTLBuffer] = [:]
    private var texture: MTLTexture?
    private var state = ExecutionState()
    private var loaded = false
    private let lock = NSLock()
    private let stride: Int

    public init(model: ProductionModel, extent: Extent, precision: Precision, candidate: MetalCandidate, diagnostic: Bool) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw QualificationError("Metal unavailable") }
        self.device = device; self.queue = queue; self.extent = extent; self.precision = precision
        self.candidate = candidate; self.diagnostic = diagnostic
        stride = ((extent.width * 8 + 255) / 256) * 256
        let source = OptimizedShaders.source(candidate: candidate, precision: precision)
        let options = MTLCompileOptions(); options.languageVersion = .version3_0; options.fastMathEnabled = false
        let library = try device.makeLibrary(source: source, options: options)
        var limits: [String: [String: Int]] = [:]
        for name in ["stem", "body", "head", "activate", "nearest", "compose"] {
            guard let function = library.makeFunction(name: name) else { throw QualificationError("Missing optimized kernel") }
            let pipeline = try device.makeComputePipelineState(function: function)
            let geometry = name == "stem" ? candidate.stem : name == "body" ? candidate.body : candidate.head
            let threads = ["stem", "body", "head"].contains(name) ? geometry.groupX * geometry.groupY : 64
            try require(threads <= pipeline.maxTotalThreadsPerThreadgroup && pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength,
                        "Candidate exceeds pipeline limits")
            pipelines[name] = pipeline
            limits[name] = ["threadExecutionWidth": pipeline.threadExecutionWidth, "maxThreads": pipeline.maxTotalThreadsPerThreadgroup,
                            "threadgroupBytes": pipeline.staticThreadgroupMemoryLength]
        }
        let packed = try PackedParameters(model: model, precision: precision)
        for (name, bytes) in packed.buffers { buffers[name] = try allocate(bytes.count, bytes: bytes) }
        let uploaded = packed.buffers.mapValues { $0 }
        let actual = Dictionary(uniqueKeysWithValues: uploaded.keys.map { name in
            (name, Data(bytes: buffers[name]!.contents(), count: buffers[name]!.length))
        })
        try packed.verify(actual, modelHash: ProductionModel.fileHash)
        let shape: [UInt32] = [UInt32(extent.width), UInt32(extent.height), diagnostic ? 1 : 0]
        buffers["shape"] = try allocate(12, bytes: shape.withUnsafeBytes { Data($0) })
        buffers["input"] = try allocate(extent.pixels * 16)
        let activationBytes = extent.pixels * 16 * precision.bytesPerElement
        buffers["ping"] = try allocate(activationBytes); buffers["pong"] = try allocate(activationBytes)
        if diagnostic {
            for name in ["stem", "body.0", "body.1"] { buffers["capture." + name] = try allocate(activationBytes) }
        }
        if !candidate.fusedHead {
            buffers["nearest"] = try allocate(activationBytes * 4)
            buffers["head"] = try allocate(extent.pixels * 4 * 16)
        }
        buffers["final"] = try allocate(diagnostic ? extent.pixels * 4 * 16 : 16)
        buffers["readback"] = try allocate(stride * extent.height * 2)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
            width: extent.width * 2, height: extent.height * 2, mipmapped: false)
        descriptor.storageMode = .private; descriptor.usage = [.shaderWrite, .shaderRead]
        guard let output = device.makeTexture(descriptor: descriptor) else { throw QualificationError("Optimized texture allocation failed") }
        texture = output
        configuration = OptimizedConfiguration(candidate: candidate, precision: precision, diagnostic: diagnostic, extent: extent,
            device: device.name, shaderSha256: sha256(Data(source.utf8)), weightHashes: packed.hashes, stem: candidate.stem,
            body: candidate.body, head: candidate.head, pipelineLimits: limits,
            configuredBufferBytes: buffers.values.reduce(0) { $0 + $1.length }, outputTextureBytes: extent.pixels * 16)
    }

    private func allocate(_ length: Int, bytes: Data? = nil) throws -> MTLBuffer {
        try require(length > 0 && length <= device.maxBufferLength, "Invalid optimized allocation")
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else { throw QualificationError("Optimized buffer allocation failed") }
        if let bytes = bytes {
            try require(bytes.count == length, "Upload size mismatch")
            bytes.withUnsafeBytes { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        }
        return buffer
    }

    public func loadInput(_ planar: [Float]) throws {
        lock.lock(); defer { lock.unlock() }
        try state.requireLive()
        try require(planar.count == extent.pixels * 3 && planar.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 }, "Invalid optimized RGB input")
        let packed = try PackedLayout.pack(planar, extent: extent, channels: 3)
        packed.withUnsafeBytes { buffers["input"]!.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        loaded = true
    }

    private func dispatch(_ name: String, buffers names: [String], command: MTLCommandBuffer,
                          geometry: KernelGeometry? = nil, width: Int, height: Int = 1, depth: Int = 1) throws {
        guard let encoder = command.makeComputeCommandEncoder() else { throw QualificationError("Optimized encoder unavailable") }
        defer { encoder.endEncoding() }
        encoder.label = name; encoder.setComputePipelineState(pipelines[name]!)
        for (index, key) in names.enumerated() { encoder.setBuffer(buffers[key], offset: 0, index: index) }
        if name == "head" || name == "compose" { encoder.setTexture(texture, index: 0) }
        if let geometry = geometry {
            let tileWidth = geometry.groupX * geometry.blockX, tileHeight = geometry.groupY * geometry.blockY
            encoder.dispatchThreadgroups(MTLSize(width: (width + tileWidth - 1) / tileWidth, height: (height + tileHeight - 1) / tileHeight, depth: depth),
                threadsPerThreadgroup: MTLSize(width: geometry.groupX, height: geometry.groupY, depth: 1))
        } else {
            encoder.dispatchThreads(MTLSize(width: width, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
    }

    private func submit(readback: Bool) throws -> MTLCommandBuffer {
        guard let command = queue.makeCommandBuffer() else { throw QualificationError("Optimized command unavailable") }
        var current = "input"
        for layer in ["stem", "body.0", "body.1"] {
            let stem = layer == "stem", destination = current == "ping" ? "pong" : "ping"
            let geometry = stem ? candidate.stem : candidate.body
            try dispatch(stem ? "stem" : "body", buffers: [current, layer + ".weight", layer + ".bias", destination, "shape"],
                command: command, geometry: geometry, width: extent.width, height: extent.height, depth: 16 / geometry.outputBlock)
            current = destination
            if !candidate.fusedActivation {
                let activated = current == "ping" ? "pong" : "ping"
                try dispatch("activate", buffers: [current, activated, "shape"], command: command, width: extent.pixels * 4)
                current = activated
            }
            if diagnostic {
                guard let blit = command.makeBlitCommandEncoder() else { throw QualificationError("Stage capture encoder unavailable") }
                blit.copy(from: buffers[current]!, sourceOffset: 0, to: buffers["capture." + layer]!, destinationOffset: 0, size: buffers[current]!.length)
                blit.endEncoding()
            }
        }
        if !candidate.fusedHead {
            try dispatch("nearest", buffers: [current, "nearest", "shape"], command: command, width: extent.pixels * 16)
            current = "nearest"
        }
        try dispatch("head", buffers: [current, "head.weight", "head.bias", candidate.fusedHead ? "final" : "head", "shape", "input"],
            command: command, geometry: candidate.head, width: extent.width * 2, height: extent.height * 2)
        if !candidate.fusedHead {
            try dispatch("compose", buffers: ["head", "input", "final", "shape"], command: command, width: extent.pixels * 4)
        }
        if readback {
            guard let blit = command.makeBlitCommandEncoder() else { throw QualificationError("RGBA capture encoder unavailable") }
            blit.copy(from: texture!, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                sourceSize: MTLSize(width: extent.width * 2, height: extent.height * 2, depth: 1), to: buffers["readback"]!, destinationOffset: 0,
                destinationBytesPerRow: stride, destinationBytesPerImage: stride * extent.height * 2)
            blit.endEncoding()
        }
        command.commit(); command.waitUntilCompleted()
        try state.complete(success: command.status == .completed && command.error == nil, message: "Optimized Metal execution failed: \(String(describing: command.error))")
        return command
    }

    public func capture() throws -> TensorCapture {
        lock.lock(); defer { lock.unlock() }
        try state.requireLive(); try require(loaded, "Optimized input not loaded")
        _ = try submit(readback: true)
        var stages: [StageTensor] = []
        if diagnostic {
            for name in ["stem", "body.0", "body.1"] {
                let buffer = buffers["capture." + name]!, count = extent.pixels * 16
                let packed: [Float]
                if precision == .f16 { packed = UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float16.self), count: count).map { Float($0) } }
                else { packed = Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count)) }
                stages.append(StageTensor(name: name, width: extent.width, height: extent.height, channels: 16,
                    values: try PackedLayout.unpack(packed, extent: extent, channels: 16)))
            }
            let pixels = extent.pixels * 4, pointer = buffers["final"]!.contents().assumingMemoryBound(to: Float.self)
            var values = [Float](repeating: 0, count: pixels * 3)
            for channel in 0..<3 { for pixel in 0..<pixels { values[channel * pixels + pixel] = pointer[pixel * 4 + channel] } }
            stages.append(StageTensor(name: "final", width: extent.width * 2, height: extent.height * 2, channels: 3, values: values))
        }
        var rgba = Data(); rgba.reserveCapacity(extent.pixels * 16)
        for row in 0..<(extent.height * 2) {
            rgba.append(buffers["readback"]!.contents().advanced(by: row * stride).assumingMemoryBound(to: UInt8.self), count: extent.width * 8)
        }
        return TensorCapture(checkpoints: stages, rgba: rgba)
    }

    public func measure(_ mode: OptimizedMeasurement, record: (MeasurementEvent) throws -> Void) throws -> OptimizedTiming {
        lock.lock(); defer { lock.unlock() }
        try state.requireLive()
        try require(loaded && !diagnostic && precision == .f16 && extent.width == 1280 && extent.height == 720, "Invalid timing configuration")
        let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
        try record(MeasurementEvent(phase: "preflight", index: 0, thermalState: thermalBefore, interval: nil))
        try require(thermalBefore == ProcessInfo.ThermalState.nominal.rawValue, "THERMAL_NOT_STARTED")
        try record(MeasurementEvent(phase: "begin", index: 0, thermalState: thermalBefore, interval: nil))
        func iteration(_ phase: String, _ index: Int) throws -> GPUInterval {
            let interval: GPUInterval = try autoreleasepool {
                let command = try submit(readback: false)
                return GPUInterval(start: command.gpuStartTime, end: command.gpuEndTime)
            }
            try record(MeasurementEvent(phase: phase, index: index, thermalState: ProcessInfo.processInfo.thermalState.rawValue, interval: interval))
            try require(interval.milliseconds != nil, "Invalid/unavailable GPU timestamp")
            return interval
        }
        for index in 0..<mode.warmups { _ = try iteration("warmup", index) }
        var samples: [GPUInterval] = []; samples.reserveCapacity(mode.samples)
        let start = ProcessInfo.processInfo.systemUptime
        for index in 0..<mode.samples { samples.append(try iteration("sample", index)) }
        let window = (ProcessInfo.processInfo.systemUptime - start) * 1000
        return OptimizedTiming(mode: mode, warmups: mode.warmups, requestedSamples: mode.samples,
            thermalStateBefore: thermalBefore, thermalStateAfter: ProcessInfo.processInfo.thermalState.rawValue,
            wholeGraph: TimingSeries(name: "optimized-whole-graph", samples: samples, windowMS: window))
    }

    public func destroy() {
        lock.lock(); defer { lock.unlock() }
        state.destroy(); buffers.removeAll(); pipelines.removeAll(); texture = nil
    }
}