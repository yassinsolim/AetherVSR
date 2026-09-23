import CoreVideo
import Foundation
import Metal

public struct VideoError: Error, CustomStringConvertible {
    public let description: String
    public init(_ description: String) { self.description = description }
}

public struct VideoColor: Codable, Equatable, Sendable {
    public let format: UInt32
    public let matrix: String?
    public let primaries: String?
    public let transfer: String?
    public let chroma: String?
    public let width: Int
    public let height: Int

    public init(_ buffer: CVPixelBuffer) throws {
        format = CVPixelBufferGetPixelFormatType(buffer)
        width = CVPixelBufferGetWidth(buffer); height = CVPixelBufferGetHeight(buffer)
        func attachment(_ key: CFString) -> String? { CVBufferCopyAttachment(buffer, key, nil) as? String }
        matrix = attachment(kCVImageBufferYCbCrMatrixKey)
        primaries = attachment(kCVImageBufferColorPrimariesKey)
        transfer = attachment(kCVImageBufferTransferFunctionKey)
        chroma = attachment(kCVImageBufferChromaLocationTopFieldKey)
        guard width > 0 && height > 0 && width <= 1280 && height <= 720 else { throw VideoError("Unsupported video extent") }
        guard [kCVPixelFormatType_32BGRA, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange].contains(format) else { throw VideoError("Unsupported pixel format: \(format)") }
        if let raw = CVBufferCopyAttachment(buffer, kCVImageBufferCleanApertureKey, nil) {
            guard let aperture = raw as? [String: NSNumber],
                  aperture[kCVImageBufferCleanApertureWidthKey as String]?.doubleValue == Double(width),
                  aperture[kCVImageBufferCleanApertureHeightKey as String]?.doubleValue == Double(height),
                  aperture[kCVImageBufferCleanApertureHorizontalOffsetKey as String]?.doubleValue == 0,
                  aperture[kCVImageBufferCleanApertureVerticalOffsetKey as String]?.doubleValue == 0 else { throw VideoError("Unsupported clean aperture") }
        }
        if let raw = CVBufferCopyAttachment(buffer, kCVImageBufferPixelAspectRatioKey, nil) {
            guard let ratio = raw as? [String: NSNumber],
                  let horizontal = ratio[kCVImageBufferPixelAspectRatioHorizontalSpacingKey as String]?.doubleValue,
                  let vertical = ratio[kCVImageBufferPixelAspectRatioVerticalSpacingKey as String]?.doubleValue,
                  horizontal.isFinite && horizontal > 0 && horizontal == vertical else { throw VideoError("Invalid source pixel aspect") }
        }
        if let fields = CVBufferCopyAttachment(buffer, kCVImageBufferFieldCountKey, nil) {
            guard let count = fields as? NSNumber, count.doubleValue == 1 else { throw VideoError("Interlaced input unsupported") }
        }
        guard CVBufferCopyAttachment(buffer, kCVImageBufferFieldDetailKey, nil) == nil else { throw VideoError("Field layout unsupported") }
        guard primaries == kCVImageBufferColorPrimaries_ITU_R_709_2 as String,
              [kCVImageBufferTransferFunction_ITU_R_709_2 as String, kCVImageBufferTransferFunction_sRGB as String].contains(transfer ?? "") else { throw VideoError("Unsupported or missing SDR primaries/transfer") }
        if format != kCVPixelFormatType_32BGRA {
            guard width % 2 == 0 && height % 2 == 0, CVPixelBufferGetPlaneCount(buffer) == 2,
                  matrix == kCVImageBufferYCbCrMatrix_ITU_R_709_2 as String,
                  chroma == kCVImageBufferChromaLocation_Left as String || chroma == kCVImageBufferChromaLocation_Center as String else { throw VideoError("Unsupported YUV metadata or planes") }
            if let bottom = CVBufferCopyAttachment(buffer, kCVImageBufferChromaLocationBottomFieldKey, nil) {
                guard bottom as? String == chroma else { throw VideoError("Conflicting field chroma") }
            }
        } else if CVPixelBufferGetPlaneCount(buffer) != 0 { throw VideoError("Invalid BGRA plane count") }
    }
}

public enum VideoResources {
    private static let lock = NSLock()
    private static var values: [String: Int] = [:]
    public static func change(_ key: String, by delta: Int) { lock.lock(); values[key, default: 0] += delta; lock.unlock() }
    public static func snapshot() -> [String: Int] {
        lock.lock(); defer { lock.unlock() }
        return Dictionary(uniqueKeysWithValues: ["leasedPixelBuffers", "liveTextureWrappers", "decodedSampleOwners"].map { ($0, values[$0, default: 0]) })
    }
}

public final class VideoLease: @unchecked Sendable {
    public let pixelBuffer: CVPixelBuffer
    public let wrappers: [CVMetalTexture]
    public let planes: [MTLTexture]
    public let color: VideoColor
    public var flippedFlags: [Bool] { wrappers.map { CVMetalTextureIsFlipped($0) } }
    init(pixelBuffer: CVPixelBuffer, wrappers: [CVMetalTexture], planes: [MTLTexture], color: VideoColor) {
        self.pixelBuffer = pixelBuffer; self.wrappers = wrappers; self.planes = planes; self.color = color
        VideoResources.change("leasedPixelBuffers", by: 1); VideoResources.change("liveTextureWrappers", by: wrappers.count)
    }
    deinit { VideoResources.change("leasedPixelBuffers", by: -1); VideoResources.change("liveTextureWrappers", by: -wrappers.count) }
}

public final class VideoBridge {
    public let device: MTLDevice
    private let cache: CVMetalTextureCache
    private let pipeline: MTLComputePipelineState

    public init(device: MTLDevice) throws {
        self.device = device
        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache) == kCVReturnSuccess, let cache = cache else { throw VideoError("Texture cache creation failed") }
        self.cache = cache
        let source = try String(contentsOf: Bundle.module.url(forResource: "Ingest", withExtension: "metal")!, encoding: .utf8)
        let options = MTLCompileOptions(); options.languageVersion = .version3_0; options.mathMode = .safe
        let library = try device.makeLibrary(source: source, options: options)
        pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "ingest")!)
    }

    public func map(_ buffer: CVPixelBuffer) throws -> VideoLease {
        let color = try VideoColor(buffer)
        let yuv = color.format != kCVPixelFormatType_32BGRA
        var wrappers: [CVMetalTexture] = [], planes: [MTLTexture] = []
        for plane in 0..<(yuv ? 2 : 1) {
            let width = yuv ? CVPixelBufferGetWidthOfPlane(buffer, plane) : color.width
            let height = yuv ? CVPixelBufferGetHeightOfPlane(buffer, plane) : color.height
            guard width == color.width / (plane == 1 ? 2 : 1), height == color.height / (plane == 1 ? 2 : 1) else { throw VideoError("Plane extent mismatch") }
            let format: MTLPixelFormat = yuv ? (plane == 0 ? .r8Unorm : .rg8Unorm) : .bgra8Unorm
            var wrapper: CVMetalTexture?
            let attributes = [kCVMetalTextureUsage as String: MTLTextureUsage.shaderRead.rawValue] as CFDictionary
            guard CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, buffer, attributes, format, width, height, plane, &wrapper) == kCVReturnSuccess,
                  let wrapper = wrapper, let texture = CVMetalTextureGetTexture(wrapper),
                  texture.width == width, texture.height == height, texture.pixelFormat == format else { throw VideoError("Core Video texture mapping failed") }
            wrappers.append(wrapper); planes.append(texture)
        }
        return VideoLease(pixelBuffer: buffer, wrappers: wrappers, planes: planes, color: color)
    }

    public func encode(_ lease: VideoLease, into input: MTLBuffer, command: MTLCommandBuffer) throws {
        guard input.device === device, command.device === device,
              input.length >= lease.color.width * lease.color.height * 16,
              let encoder = command.makeComputeCommandEncoder() else { throw VideoError("Invalid ingest destination") }
        encoder.label = "Core Video SDR ingest"; encoder.setComputePipelineState(pipeline)
        encoder.setTexture(lease.planes[0], index: 0); encoder.setTexture(lease.planes.last!, index: 1)
        encoder.setBuffer(input, offset: 0, index: 0)
        let format: UInt32 = lease.color.format == kCVPixelFormatType_32BGRA ? 0 : lease.color.format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ? 1 : 2
        var parameters: [UInt32] = [UInt32(lease.color.width), UInt32(lease.color.height), format, lease.color.chroma == kCVImageBufferChromaLocation_Center as String ? 1 : 0]
        parameters.withUnsafeMutableBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 1) }
        encoder.dispatchThreads(MTLSize(width: lease.color.width, height: lease.color.height, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        encoder.endEncoding()
    }

    public func flushAfterDrain() { CVMetalTextureCacheFlush(cache, 0) }
}