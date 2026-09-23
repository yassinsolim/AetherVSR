import Foundation
import Metal

public final class FrameNetwork {
    public let device: MTLDevice
    public let extent: Extent
    public let precision: Precision
    public let input: MTLBuffer
    public let output: MTLTexture
    public let shaderSha256: String
    public let packedWeightHashes: [String: String]
    public let allocatedBytes: Int
    private let diagnostic: Bool
    private let pipelines: [String: MTLComputePipelineState]
    private let weights: [String: MTLBuffer]
    private let shape: MTLBuffer
    private let ping: MTLBuffer
    private let pong: MTLBuffer
    private let final: MTLBuffer
    private let captures: [MTLBuffer]
    private let validationFlags: MTLBuffer
    private let validationCount: MTLBuffer
    private let hiddenValidation: MTLComputePipelineState
    private let finalValidation: MTLComputePipelineState

    public init(device: MTLDevice, model: ProductionModel, extent: Extent, precision: Precision = .f16, diagnostic: Bool = false) throws {
        self.device = device; self.extent = extent; self.precision = precision; self.diagnostic = diagnostic
        var bytesAllocated = 0
        func buffer(_ length: Int, data: Data? = nil, shared: Bool = false) throws -> MTLBuffer {
            try require(length > 0 && length <= device.maxBufferLength, "Frame network buffer exceeds device limit")
            guard let result = device.makeBuffer(length: length, options: shared || data != nil ? .storageModeShared : .storageModePrivate) else { throw QualificationError("Frame network allocation failed") }
            if let data = data {
                try require(length == data.count, "Frame network upload mismatch")
                data.withUnsafeBytes { result.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            }
            bytesAllocated += length; return result
        }
        let source = OptimizedShaders.source(candidate: .f, precision: precision)
        shaderSha256 = sha256(Data(source.utf8))
        let options = MTLCompileOptions(); options.languageVersion = .version3_0
        options.mathMode = .safe; options.mathFloatingPointFunctions = .precise
        let library = try device.makeLibrary(source: source, options: options)
        var compiled: [String: MTLComputePipelineState] = [:]
        for name in ["stem", "body", "head"] {
            let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: name)!)
            let geometry = name == "stem" ? MetalCandidate.f.stem : name == "body" ? MetalCandidate.f.body : MetalCandidate.f.head
            try require(geometry.groupX * geometry.groupY <= pipeline.maxTotalThreadsPerThreadgroup && pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength, "Frame pipeline limit")
            compiled[name] = pipeline
        }
        pipelines = compiled
        let scalar = precision == .f16 ? "half" : "float"
        let auditSource = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void hidden(device const \(scalar)4* values [[buffer(0)]], device atomic_uint* flags [[buffer(1)]], constant uint& count [[buffer(2)]], uint index [[thread_position_in_grid]]) {
            if(index>=count) return;
            float4 value=float4(values[index]);
            if(!all(isfinite(value)) || any(value < -1.0f) || any(value > 1.0f)) atomic_fetch_or_explicit(flags,1u,memory_order_relaxed);
        }
        kernel void finalCheck(device const float4* values [[buffer(0)]], device atomic_uint* flags [[buffer(1)]], constant uint& count [[buffer(2)]], uint index [[thread_position_in_grid]]) {
            if(index>=count) return;
            float4 value=values[index];
            if(!all(isfinite(value)) || any(value < 0.0f) || any(value > 1.0f) || value.a != 1.0f) atomic_fetch_or_explicit(flags,2u,memory_order_relaxed);
        }
        """
        let auditLibrary = try device.makeLibrary(source: auditSource, options: options)
        hiddenValidation = try device.makeComputePipelineState(function: auditLibrary.makeFunction(name: "hidden")!)
        finalValidation = try device.makeComputePipelineState(function: auditLibrary.makeFunction(name: "finalCheck")!)
        let parameters = try PackedParameters(model: model, precision: precision)
        packedWeightHashes = parameters.hashes
        var weightBuffers: [String: MTLBuffer] = [:]
        for (name, data) in parameters.buffers { weightBuffers[name] = try buffer(data.count, data: data) }
        try parameters.verify(weightBuffers.mapValues { Data(bytes: $0.contents(), count: $0.length) }, modelHash: ProductionModel.fileHash)
        weights = weightBuffers
        let dimensions: [UInt32] = [UInt32(extent.width), UInt32(extent.height), 1]
        shape = try buffer(12, data: dimensions.withUnsafeBytes { Data($0) })
        input = try buffer(extent.pixels * 16, shared: diagnostic)
        let hiddenBytes = extent.pixels * 16 * precision.bytesPerElement
        ping = try buffer(hiddenBytes); pong = try buffer(hiddenBytes)
        final = try buffer(extent.pixels * 64, shared: diagnostic)
        validationFlags = try buffer(4, shared: true)
        validationCount = try buffer(4, data: [UInt32(extent.pixels * 4)].withUnsafeBytes { Data($0) })
        captures = diagnostic ? try (0..<3).map { _ in try buffer(hiddenBytes, shared: true) } : []
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: extent.width * 2, height: extent.height * 2, mipmapped: false)
        descriptor.storageMode = .private; descriptor.usage = [.shaderRead, .shaderWrite]
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw QualificationError("Frame output texture allocation failed") }
        output = texture; allocatedBytes = bytesAllocated
    }

    public func encode(command: MTLCommandBuffer) throws {
        try require(command.device === device, "Frame command belongs to another device")
        guard let reset = command.makeBlitCommandEncoder() else { throw QualificationError("Frame validation reset unavailable") }
        reset.fill(buffer: validationFlags, range: 0..<4, value: 0); reset.endEncoding()
        for (index, layer) in ["stem", "body.0", "body.1", "head"].enumerated() {
            let kernel = index == 0 ? "stem" : index == 3 ? "head" : "body"
            let geometry = index == 0 ? MetalCandidate.f.stem : index == 3 ? MetalCandidate.f.head : MetalCandidate.f.body
            let source = index == 0 ? input : index == 2 ? pong : ping
            let destination = index == 3 ? final : index == 1 ? pong : ping
            guard let encoder = command.makeComputeCommandEncoder() else { throw QualificationError("Frame compute encoder unavailable") }
            encoder.label = "F.\(layer)"; encoder.setComputePipelineState(pipelines[kernel]!)
            for (binding, resource) in [source, weights[layer + ".weight"]!, weights[layer + ".bias"]!, destination, shape].enumerated() {
                encoder.setBuffer(resource, offset: 0, index: binding)
            }
            if index == 3 { encoder.setBuffer(input, offset: 0, index: 5); encoder.setTexture(output, index: 0) }
            let width = extent.width * (index == 3 ? 2 : 1), height = extent.height * (index == 3 ? 2 : 1)
            encoder.dispatchThreadgroups(MTLSize(width: (width + geometry.groupX * geometry.blockX - 1) / (geometry.groupX * geometry.blockX),
                height: (height + geometry.groupY * geometry.blockY - 1) / (geometry.groupY * geometry.blockY), depth: 1),
                threadsPerThreadgroup: MTLSize(width: geometry.groupX, height: geometry.groupY, depth: 1))
            encoder.endEncoding()
            guard let audit = command.makeComputeCommandEncoder() else { throw QualificationError("Frame output validation unavailable") }
            audit.label = "F output finite/range validation"
            audit.setComputePipelineState(index == 3 ? finalValidation : hiddenValidation)
            audit.setBuffer(destination, offset: 0, index: 0); audit.setBuffer(validationFlags, offset: 0, index: 1)
            audit.setBuffer(validationCount, offset: 0, index: 2)
            audit.dispatchThreads(MTLSize(width: extent.pixels * 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            audit.endEncoding()
            if diagnostic && index < 3 {
                guard let copy = command.makeBlitCommandEncoder() else { throw QualificationError("Frame diagnostic copy unavailable") }
                copy.copy(from: destination, sourceOffset: 0, to: captures[index], destinationOffset: 0, size: destination.length); copy.endEncoding()
            }
        }
    }

    public func validationFlagsAfterCompletion() -> UInt32 { validationFlags.contents().assumingMemoryBound(to: UInt32.self).pointee }

    public func diagnosticStagesAfterCompletion() throws -> [StageTensor] {
        try require(diagnostic, "Diagnostic frame buffers not configured")
        var result: [StageTensor] = []
        for (index, name) in ["stem", "body.0", "body.1"].enumerated() {
            let count = extent.pixels * 16, pointer = captures[index].contents()
            let packed = precision == .f16
                ? UnsafeBufferPointer(start: pointer.assumingMemoryBound(to: Float16.self), count: count).map { Float($0) }
                : Array(UnsafeBufferPointer(start: pointer.assumingMemoryBound(to: Float.self), count: count))
            result.append(StageTensor(name: name, width: extent.width, height: extent.height, channels: 16,
                values: try PackedLayout.unpack(packed, extent: extent, channels: 16)))
        }
        let pointer = final.contents().assumingMemoryBound(to: Float.self), pixels = extent.pixels * 4
        var planar = [Float](repeating: 0, count: pixels * 3)
        for channel in 0..<3 { for pixel in 0..<pixels { planar[channel * pixels + pixel] = pointer[pixel * 4 + channel] } }
        result.append(StageTensor(name: "final", width: extent.width * 2, height: extent.height * 2, channels: 3, values: planar))
        return result
    }
}