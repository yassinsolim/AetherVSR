import Foundation
import Metal

private struct ConvolutionShape {
    var width: UInt32, height: UInt32, inputChannels: UInt32, outputChannels: UInt32, kernel: UInt32
}
private struct TensorShape { var width: UInt32, height: UInt32, channels: UInt32 }

public struct TensorCapture {
    public let checkpoints: [StageTensor]
    public let rgba: Data
}

public struct GPUInterval: Encodable {
    public let startSeconds: Double?
    public let endSeconds: Double?
    public var milliseconds: Double? {
        guard let start = startSeconds, let end = endSeconds else { return nil }
        return (end - start) * 1000
    }
    init(start: Double, end: Double) {
        let valid = start.isFinite && end.isFinite && start > 0 && end > start && ((end - start) * 1000).isFinite
        startSeconds = valid ? start : nil; endSeconds = valid ? end : nil
    }
    private enum CodingKeys: String, CodingKey { case startSeconds, endSeconds, milliseconds }
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(startSeconds, forKey: .startSeconds)
        try container.encode(endSeconds, forKey: .endSeconds)
        try container.encode(milliseconds, forKey: .milliseconds)
    }
}

public struct TimingSeries: Encodable {
    public let name: String
    public let samples: [GPUInterval]
    public let observationWindowMS: Double
    public let measuredSamples: Int
    public let statisticsMS: [String: Double?]
    public let status: String
    public var complete: Bool { !samples.isEmpty && measuredSamples == samples.count }

    init(name: String, samples: [GPUInterval], windowMS: Double) {
        self.name = name; self.samples = samples; observationWindowMS = windowMS
        let sorted = samples.compactMap(\.milliseconds).sorted()
        measuredSamples = sorted.count
        status = sorted.isEmpty ? "not measured" : sorted.count == samples.count ? "measured" : "partially measured"
        func percentile(_ fraction: Double) -> Double? {
            guard !sorted.isEmpty else { return nil }
            let position = Double(sorted.count - 1) * fraction
            let lower = Int(position), upper = min(lower + 1, sorted.count - 1)
            return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - Double(lower))
        }
        statisticsMS = ["p50": percentile(0.5), "p95": percentile(0.95), "max": sorted.last]
    }
}

public struct TimingCapture: Encodable {
    public let warmupIterations = 10
    public let measuredIterations = 60
    public let wholeGraph: TimingSeries
    public let isolatedStages: [TimingSeries]
    public let configuredBufferBytes: Int
    public var complete: Bool { wholeGraph.complete && isolatedStages.allSatisfy(\.complete) }
}

struct ExecutionState {
    private var failed = false
    private var destroyed = false
    func requireLive() throws { try require(!failed && !destroyed, "Engine failed or destroyed") }
    mutating func complete(success: Bool, message: String) throws {
        if !success { failed = true; throw QualificationError(message) }
        try requireLive()
    }
    mutating func destroy() { destroyed = true }
}

public final class MetalEngine {
    public let device: MTLDevice
    public let extent: Extent
    public let precision: Precision
    private let queue: MTLCommandQueue
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private var buffers: [String: MTLBuffer] = [:]
    private var texture: MTLTexture?
    private let lock = NSLock()
    private var state = ExecutionState()
    private var inputLoaded = false
    private let outputStride: Int
    public let stageOrder = ["stem.conv", "stem", "body.0.conv", "body.0", "body.1.conv", "body.1",
                             "nearest.features", "head", "nearest.input", "residual", "final", "rgba"]

    public init(model: ProductionModel, extent: Extent, precision: Precision) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw QualificationError("Metal device/queue unavailable")
        }
        self.device = device; self.queue = queue; self.extent = extent; self.precision = precision
        outputStride = ((extent.width * 2 * 4 + 255) / 256) * 256
        let options = MTLCompileOptions()
        options.languageVersion = .version3_0
        options.fastMathEnabled = false
        let library = try device.makeLibrary(source: Self.shaderSource(), options: options)
        for name in ["convolution32", "activation32", "nearest32", "residual32", "clamp32", "rgba32",
                 "convolution16", "activation16", "nearest16", "residual16"] {
            guard let function = library.makeFunction(name: name) else { throw QualificationError("Missing Metal kernel: \(name)") }
            let pipeline = try device.makeComputePipelineState(function: function)
            try require(pipeline.maxTotalThreadsPerThreadgroup >= 64, "Unsupported threadgroup limit")
            pipelines[name] = pipeline
        }
        let elementBytes = precision.bytesPerElement
        for (name, values) in model.weights {
            let buffer = try allocate(values.count * elementBytes)
            if precision == .f16 {
                let half = values.map { Float16($0) }
                try require(half.allSatisfy(\.isFinite), "Weight overflow in f16")
                half.withUnsafeBytes { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            } else {
                values.withUnsafeBytes { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            }
            buffers[name] = buffer
        }
        buffers["input"] = try allocate(extent.pixels * 3 * 4)
        if precision == .f16 { buffers["input.half"] = try allocate(extent.pixels * 3 * 2) }
        buffers["params.stem"] = try parameter(ConvolutionShape(width: UInt32(extent.width), height: UInt32(extent.height),
            inputChannels: 3, outputChannels: 16, kernel: 5))
        buffers["params.body"] = try parameter(ConvolutionShape(width: UInt32(extent.width), height: UInt32(extent.height),
            inputChannels: 16, outputChannels: 16, kernel: 3))
        buffers["params.head"] = try parameter(ConvolutionShape(width: UInt32(extent.width * 2), height: UInt32(extent.height * 2),
            inputChannels: 16, outputChannels: 3, kernel: 3))
        buffers["params.nearest.features"] = try parameter(TensorShape(width: UInt32(extent.width), height: UInt32(extent.height), channels: 16))
        buffers["params.nearest.input"] = try parameter(TensorShape(width: UInt32(extent.width), height: UInt32(extent.height), channels: 3))
        buffers["params.rgba"] = try parameter(TensorShape(width: UInt32(extent.width * 2), height: UInt32(extent.height * 2), channels: 3))
        buffers["params.hiddenCount"] = try parameter(UInt32(extent.pixels * 16))
        buffers["params.outputCount"] = try parameter(UInt32(extent.pixels * 12))
        for name in ["stem.conv", "stem", "body.0.conv", "body.0", "body.1.conv", "body.1"] {
            buffers[name] = try allocate(extent.pixels * 16 * elementBytes)
        }
        buffers["nearest.features"] = try allocate(extent.pixels * 4 * 16 * elementBytes)
        buffers["head"] = try allocate(extent.pixels * 4 * 3 * elementBytes)
        for name in ["nearest.input", "residual", "final"] {
            buffers[name] = try allocate(extent.pixels * 4 * 3 * 4)
        }
        buffers["rgba.readback"] = try allocate(outputStride * extent.height * 2)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
            width: extent.width * 2, height: extent.height * 2, mipmapped: false)
        descriptor.storageMode = .private; descriptor.usage = [.shaderWrite, .shaderRead]
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw QualificationError("Output texture allocation failed") }
        self.texture = texture
    }

    static func shaderSource() throws -> String {
        guard let resource = Bundle.module.url(forResource: "Shaders", withExtension: "metal") else {
            throw QualificationError("Missing bundled Metal source")
        }
        return try String(contentsOf: resource, encoding: .utf8)
    }

    private func allocate(_ length: Int) throws -> MTLBuffer {
        try require(length > 0 && length <= device.maxBufferLength, "Buffer exceeds device allocation limit")
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw QualificationError("Metal buffer allocation failed")
        }
        return buffer
    }

    private func parameter<Value>(_ value: Value) throws -> MTLBuffer {
        var value = value
        let buffer = try allocate(MemoryLayout<Value>.stride)
        withUnsafeBytes(of: &value) { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return buffer
    }

    public func loadInput(_ input: [Float]) throws {
        lock.lock(); defer { lock.unlock() }
        try state.requireLive()
        try require(input.count == extent.pixels * 3 && input.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 },
                    "Invalid CHW RGB input tensor")
        input.withUnsafeBytes { buffers["input"]!.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        if precision == .f16 {
            input.map { Float16($0) }.withUnsafeBytes {
                buffers["input.half"]!.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
            }
        }
        inputLoaded = true
    }

    private func encode(_ name: String, command: MTLCommandBuffer) throws {
        guard let encoder = command.makeComputeCommandEncoder() else { throw QualificationError("Metal encoder unavailable") }
        defer { encoder.endEncoding() }
        encoder.label = name
        let pixels = extent.pixels
        let suffix = precision == .f16 ? "16" : "32"
        var kernel = "", length = pixels * 16
        switch name {
        case "stem.conv", "body.0.conv", "body.1.conv", "head":
            let stem = name == "stem.conv", head = name == "head"
            let source = stem ? (precision == .f16 ? "input.half" : "input") : head ? "nearest.features" : name == "body.0.conv" ? "stem" : "body.0"
            let layer = name.replacingOccurrences(of: ".conv", with: "")
            kernel = "convolution" + suffix
            encoder.setBuffer(buffers[source], offset: 0, index: 0)
            encoder.setBuffer(buffers[layer + ".weight"], offset: 0, index: 1)
            encoder.setBuffer(buffers[layer + ".bias"], offset: 0, index: 2)
            encoder.setBuffer(buffers[name], offset: 0, index: 3)
            encoder.setBuffer(buffers[stem ? "params.stem" : head ? "params.head" : "params.body"], offset: 0, index: 4)
            length = pixels * (head ? 12 : 16)
        case "stem", "body.0", "body.1":
            kernel = "activation" + suffix
            encoder.setBuffer(buffers[name + ".conv"], offset: 0, index: 0)
            encoder.setBuffer(buffers[name], offset: 0, index: 1)
            encoder.setBuffer(buffers["params.hiddenCount"], offset: 0, index: 2)
        case "nearest.features", "nearest.input":
            kernel = "nearest" + (name == "nearest.input" ? "32" : suffix)
            let channels = name == "nearest.features" ? 16 : 3
            encoder.setBuffer(buffers[channels == 16 ? "body.1" : "input"], offset: 0, index: 0)
            encoder.setBuffer(buffers[name], offset: 0, index: 1)
            encoder.setBuffer(buffers["params." + name], offset: 0, index: 2)
            length = pixels * 4 * channels
        case "residual":
            kernel = "residual" + suffix; length = pixels * 12
            for (index, buffer) in ["head", "nearest.input", "residual"].enumerated() { encoder.setBuffer(buffers[buffer], offset: 0, index: index) }
            encoder.setBuffer(buffers["params.outputCount"], offset: 0, index: 3)
        case "final":
            kernel = "clamp32"; length = pixels * 12
            encoder.setBuffer(buffers["residual"], offset: 0, index: 0); encoder.setBuffer(buffers["final"], offset: 0, index: 1)
            encoder.setBuffer(buffers["params.outputCount"], offset: 0, index: 2)
        case "rgba":
            kernel = "rgba32"; length = pixels * 4
            encoder.setBuffer(buffers["final"], offset: 0, index: 0); encoder.setTexture(texture, index: 0)
            encoder.setBuffer(buffers["params.rgba"], offset: 0, index: 1)
        default: throw QualificationError("Unknown Metal stage")
        }
        encoder.setComputePipelineState(pipelines[kernel]!)
        encoder.dispatchThreads(MTLSize(width: length, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    private func submit(readback: Bool, stages: [String]? = nil) throws -> MTLCommandBuffer {
        guard let command = queue.makeCommandBuffer() else { throw QualificationError("Metal command buffer unavailable") }
        for name in stages ?? stageOrder { try encode(name, command: command) }
        if readback {
            guard let blit = command.makeBlitCommandEncoder() else { throw QualificationError("Metal readback encoder unavailable") }
            blit.copy(from: texture!, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                      sourceSize: MTLSize(width: extent.width * 2, height: extent.height * 2, depth: 1),
                      to: buffers["rgba.readback"]!, destinationOffset: 0, destinationBytesPerRow: outputStride,
                      destinationBytesPerImage: outputStride * extent.height * 2)
            blit.endEncoding()
        }
        command.commit(); command.waitUntilCompleted()
        try state.complete(success: command.status == .completed && command.error == nil,
                   message: "Metal execution failed: \(String(describing: command.error))")
        return command
    }

    public func benchmark() throws -> TimingCapture {
        lock.lock(); defer { lock.unlock() }
        try state.requireLive()
        try require(inputLoaded && extent.width == 1280 && extent.height == 720, "Benchmark requires configured 1280x720 input")
        func interval(_ stages: [String]? = nil) throws -> GPUInterval {
            try autoreleasepool {
                let command = try submit(readback: false, stages: stages)
                return GPUInterval(start: command.gpuStartTime, end: command.gpuEndTime)
            }
        }
        for _ in 0..<10 { _ = try interval() }
        let wholeStart = ProcessInfo.processInfo.systemUptime
        var whole: [GPUInterval] = []; whole.reserveCapacity(60)
        for _ in 0..<60 { whole.append(try interval()) }
        let wholeWindow = (ProcessInfo.processInfo.systemUptime - wholeStart) * 1000
        for _ in 0..<10 { for stage in stageOrder { _ = try interval([stage]) } }
        var isolated = stageOrder.map { _ in [GPUInterval]() }
        for index in isolated.indices { isolated[index].reserveCapacity(60) }
        let isolatedStart = ProcessInfo.processInfo.systemUptime
        for _ in 0..<60 {
            for (index, stage) in stageOrder.enumerated() { isolated[index].append(try interval([stage])) }
        }
        let isolatedWindow = (ProcessInfo.processInfo.systemUptime - isolatedStart) * 1000
        return TimingCapture(wholeGraph: TimingSeries(name: "whole-graph", samples: whole, windowMS: wholeWindow),
            isolatedStages: stageOrder.enumerated().map { TimingSeries(name: $0.element, samples: isolated[$0.offset], windowMS: isolatedWindow) },
            configuredBufferBytes: buffers.values.reduce(0) { $0 + $1.length })
    }

    public func capture() throws -> TensorCapture {
        lock.lock(); defer { lock.unlock() }
        try state.requireLive()
        try require(inputLoaded, "Input not loaded")
        _ = try submit(readback: true)
        let checkpoints = ["stem", "body.0", "body.1", "final"].map { name -> StageTensor in
            let final = name == "final", width = extent.width * (final ? 2 : 1), height = extent.height * (final ? 2 : 1), channels = final ? 3 : 16
            let count = width * height * channels
            let values: [Float]
            if precision == .f16 && !final {
                let pointer = buffers[name]!.contents().bindMemory(to: Float16.self, capacity: count)
                values = UnsafeBufferPointer(start: pointer, count: count).map { Float($0) }
            } else {
                let pointer = buffers[name]!.contents().bindMemory(to: Float.self, capacity: count)
                values = Array(UnsafeBufferPointer(start: pointer, count: count))
            }
            return StageTensor(name: name, width: width, height: height, channels: channels,
                               values: values)
        }
        var rgba = Data()
        for row in 0..<(extent.height * 2) {
            rgba.append(buffers["rgba.readback"]!.contents().advanced(by: row * outputStride).assumingMemoryBound(to: UInt8.self), count: extent.width * 8)
        }
        return TensorCapture(checkpoints: checkpoints, rgba: rgba)
    }

    public func destroy() {
        lock.lock(); defer { lock.unlock() }
        buffers.removeAll(); pipelines.removeAll(); texture = nil; state.destroy()
    }
}