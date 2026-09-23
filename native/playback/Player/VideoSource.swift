import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import NativeVideo

public final class DecodedFrame: @unchecked Sendable {
    public let sample: AVPlayerVideoOutput.Sample
    public let readOnlyBuffer: CVReadOnlyPixelBuffer
    public let pixelBuffer: CVPixelBuffer
    public let pts: Double
    public let configuration: ObjectIdentifier
    public let item: AVPlayerItem
    init(sample: AVPlayerVideoOutput.Sample, buffer: CVReadOnlyPixelBuffer, pixelBuffer: CVPixelBuffer, item: AVPlayerItem) {
        self.sample = sample; readOnlyBuffer = buffer; self.pixelBuffer = pixelBuffer
        self.item = item
        pts = sample.presentationTime.seconds; configuration = ObjectIdentifier(sample.activeConfiguration)
        VideoResources.change("decodedSampleOwners", by: 1)
    }
    deinit { VideoResources.change("decodedSampleOwners", by: -1) }
}

@MainActor public final class VideoSource {
    public let player = AVQueuePlayer()
    public private(set) var output: AVPlayerVideoOutput?
    public private(set) var looper: AVPlayerLooper?
    public private(set) var duration: Double = 0
    public private(set) var sourceFPS: Double = 0
    public private(set) var url: URL?
    public private(set) var sourceGeneration: UInt64 = 0
    public private(set) var hasAudio = false

    public init() { player.preventsDisplaySleepDuringVideoPlayback = true }

    public func open(_ url: URL) async throws {
        guard url.isFileURL && url.pathExtension.lowercased() == "mp4" else { throw VideoError("Only local progressive MP4 is supported") }
        stop(); let generation = sourceGeneration
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let track = tracks.first else { throw VideoError("No video track") }
        let size = try await track.load(.naturalSize), transform = try await track.load(.preferredTransform)
        let rate = try await track.load(.nominalFrameRate), duration = try await asset.load(.duration)
        let descriptions = try await track.load(.formatDescriptions)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        guard generation == sourceGeneration else { throw CancellationError() }
        guard size.width == 1280 && size.height == 720 && transform == .identity,
              rate > 0 && rate <= 60.1, duration.seconds.isFinite && duration.seconds > 0,
              descriptions.allSatisfy({ CMFormatDescriptionGetMediaSubType($0) == kCMVideoCodecType_H264 }) else { throw VideoError("Unsupported local video geometry/codec/rate") }
        self.url = url; self.duration = duration.seconds; sourceFPS = Double(rate); hasAudio = !audioTracks.isEmpty
        let item = AVPlayerItem(asset: asset)
        looper = AVPlayerLooper(player: player, templateItem: item)
        attachOutput()
    }

    public func attachOutput() {
        player.videoOutput = nil
        let specification = AVVideoOutputSpecification(tagCollections: [[.mediaType(.video)]])
        specification.defaultOutputSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelBufferMetalCompatibilityKey as String: true]
        let output = AVPlayerVideoOutput(specification: specification)
        player.videoOutput = output; self.output = output
    }

    public func sample(hostTime: Double) throws -> DecodedFrame? {
        guard let sample = output?.sample(forHostTime: CMTime(seconds: hostTime, preferredTimescale: 1_000_000_000)) else { return nil }
          guard sample.taggedBuffers.count == 1, sample.activeConfiguration.preferredTransform == .identity,
              case .pixelBuffer(let buffer) = sample.taggedBuffers[0].content else { throw VideoError("Unsupported tagged video sample") }
          guard let item = sample.activeConfiguration.sourcePlayerItem else { return nil }
          return buffer.withUnsafeBuffer { DecodedFrame(sample: sample, buffer: buffer, pixelBuffer: $0, item: item) }
    }

    public func stop() {
        sourceGeneration &+= 1; player.pause(); player.videoOutput = nil; output = nil
        looper?.disableLooping(); looper = nil; player.removeAllItems(); url = nil; duration = 0; sourceFPS = 0; hasAudio = false
    }
}