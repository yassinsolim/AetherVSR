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
        var count = UInt32(pixels * 16), kernel = "", length = pixels * 16
        switch name {
        case "stem.conv", "body.0.conv", "body.1.conv", "head":
            let stem = name == "stem.conv", head = name == "head"
            let source = stem ? (precision == .f16 ? "input.half" : "input") : head ? "nearest.features" : name == "body.0.conv" ? "stem" : "body.0"
            let layer = name.replacingOccurrences(of: ".conv", with: "")
            kernel = "convolution" + suffix
            var shape = ConvolutionShape(width: UInt32(extent.width * (head ? 2 : 1)), height: UInt32(extent.height * (head ? 2 : 1)),
                                         inputChannels: UInt32(stem ? 3 : 16), outputChannels: UInt32(head ? 3 : 16), kernel: UInt32(stem ? 5 : 3))
            encoder.setBuffer(buffers[source], offset: 0, index: 0)
            encoder.setBuffer(buffers[layer + ".weight"], offset: 0, index: 1)
            encoder.setBuffer(buffers[layer + ".bias"], offset: 0, index: 2)
            encoder.setBuffer(buffers[name], offset: 0, index: 3)
            encoder.setBytes(&shape, length: MemoryLayout<ConvolutionShape>.stride, index: 4)
            length = Int(shape.width * shape.height * shape.outputChannels)
        case "stem", "body.0", "body.1":
            kernel = "activation" + suffix
            encoder.setBuffer(buffers[name + ".conv"], offset: 0, index: 0)
            encoder.setBuffer(buffers[name], offset: 0, index: 1)
            encoder.setBytes(&count, length: 4, index: 2)
        case "nearest.features", "nearest.input":
            kernel = "nearest" + (name == "nearest.input" ? "32" : suffix)
            let channels = name == "nearest.features" ? 16 : 3
            var shape = TensorShape(width: UInt32(extent.width), height: UInt32(extent.height), channels: UInt32(channels))
            encoder.setBuffer(buffers[channels == 16 ? "body.1" : "input"], offset: 0, index: 0)
            encoder.setBuffer(buffers[name], offset: 0, index: 1)
            encoder.setBytes(&shape, length: MemoryLayout<TensorShape>.stride, index: 2)
            length = pixels * 4 * channels
        case "residual":
            kernel = "residual" + suffix; count = UInt32(pixels * 12); length = Int(count)
            for (index, buffer) in ["head", "nearest.input", "residual"].enumerated() { encoder.setBuffer(buffers[buffer], offset: 0, index: index) }
            encoder.setBytes(&count, length: 4, index: 3)
        case "final":
            kernel = "clamp32"; count = UInt32(pixels * 12); length = Int(count)
            encoder.setBuffer(buffers["residual"], offset: 0, index: 0); encoder.setBuffer(buffers["final"], offset: 0, index: 1)
            encoder.setBytes(&count, length: 4, index: 2)
        case "rgba":
            kernel = "rgba32"; length = pixels * 4
            var shape = TensorShape(width: UInt32(extent.width * 2), height: UInt32(extent.height * 2), channels: 3)
            encoder.setBuffer(buffers["final"], offset: 0, index: 0); encoder.setTexture(texture, index: 0)
            encoder.setBytes(&shape, length: MemoryLayout<TensorShape>.stride, index: 1)
        default: throw QualificationError("Unknown Metal stage")
        }
        encoder.setComputePipelineState(pipelines[kernel]!)
        encoder.dispatchThreads(MTLSize(width: length, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    private func submit(readback: Bool) throws -> MTLCommandBuffer {
        guard let command = queue.makeCommandBuffer() else { throw QualificationError("Metal command buffer unavailable") }
        for name in stageOrder { try encode(name, command: command) }
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