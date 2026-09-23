import AetherMetal
import AppKit
import AVFoundation
import CoreVideo
import Foundation
import MetalKit
import NativeVideo
import QuartzCore

@MainActor final class SeekCompletion {
    private var completion: ((Bool) -> Void)?
    init(_ completion: @escaping (Bool) -> Void) { self.completion = completion }
    func finish(_ success: Bool) { let callback = completion; completion = nil; callback?(success) }
}

@MainActor private final class FrameSlot {
    let network: FrameNetwork
    var identity: FrameIdentity?
    var item: AVPlayerItem?
    var frame: DecodedFrame?
    var lease: VideoLease?
    var processing = false
    var presenting = false
    var ready = false
    var current = false
    var recorded = false
    var acquisitionHost = 0.0
    var submissionHost = 0.0
    var completionHost = 0.0
    var ingestMS: Double?
    var neuralMS: Double?
    var processingMS: Double?
    var ingestStart = 0.0
    var ingestEnd = 0.0
    var networkStart = 0.0
    var networkEnd = 0.0
    var invalidOutputFlags: UInt32?
    init(_ network: FrameNetwork) { self.network = network }
}

@MainActor public final class PlaybackController: NSObject, CAMetalDisplayLinkDelegate {
    public let source = VideoSource()
    public let device: MTLDevice
    public private(set) var state = FrameState()
    public private(set) var neural = true
    public private(set) var error: String?
    public private(set) var acquired: UInt64 = 0
    public private(set) var submitted: UInt64 = 0
    public private(set) var completed: UInt64 = 0
    public private(set) var presented: UInt64 = 0
    public private(set) var busy: UInt64 = 0
    public private(set) var duplicate: UInt64 = 0
    public private(set) var opportunities: UInt64 = 0
    public private(set) var gaps: UInt64 = 0
    public private(set) var loops: UInt64 = 0
    public private(set) var lastPresentedDrawable = CGSize.zero
    public private(set) var lastPresentedOutput = CGSize.zero
    public var isSeeking: Bool { seeking || seekTarget != nil }
    public var surfaceCovered: Bool { !cover.isHidden }
    public var playbackIntended: Bool { desiredPlaying }
    public var onEvent: (([String: Any]) -> Void)?
    public var onStatus: (() -> Void)?
    private let queue: MTLCommandQueue
    private var bridge: VideoBridge?
    private let render: MTLRenderPipelineState
    private let baseline: MTLComputePipelineState
    private var slots: [FrameSlot]
    private weak var view: MTKView?
    private var link: CAMetalDisplayLink?
    private var lastConfiguration: ObjectIdentifier?
    private var previousPTS: Double?
    private var refresh = false
    private var seeking = false
    private var seekTarget: Double?
    private var desiredPlaying = false
    private var pendingSeekResume = false
    private var requestID: UInt64 = 0
    private var currentItem: ObjectIdentifier?
    private var disposed = false
    private let cover = CALayer()
    private var drainWaiters: [CheckedContinuation<Void, Never>] = []
    private var lastOpportunityHost: Double = 0

    public func diagnosticSnapshot() -> [String: Any] {
        ["rate": source.player.rate, "timeControl": source.player.timeControlStatus.rawValue, "playerTime": source.player.currentTime().seconds,
         "sourceFPS": source.sourceFPS, "sourceDuration": source.duration, "hasAudio": source.hasAudio,
         "generation": state.generation, "resources": resources(), "thermal": ProcessInfo.processInfo.thermalState.rawValue,
         "windowKey": view?.window?.isKeyWindow ?? false, "windowVisible": view?.window?.isVisible ?? false,
         "occlusionVisible": view?.window?.occlusionState.contains(.visible) ?? false]
    }

    public init(model: ProductionModel, device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw VideoError("Command queue unavailable") }
        self.queue = queue; bridge = try VideoBridge(device: device)
        let extent = try Extent(width: 1280, height: 720)
        slots = try (0..<2).map { _ in FrameSlot(try FrameNetwork(device: device, model: model, extent: extent)) }
        let source = try String(contentsOf: Bundle.module.url(forResource: "Present", withExtension: "metal")!, encoding: .utf8)
        let options = MTLCompileOptions(); options.mathMode = .safe; options.mathFloatingPointFunctions = .precise
        let library = try device.makeLibrary(source: source, options: options)
        let descriptor = MTLRenderPipelineDescriptor(); descriptor.vertexFunction = library.makeFunction(name: "vertexMain")
        descriptor.fragmentFunction = library.makeFunction(name: "fragmentMain"); descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        render = try device.makeRenderPipelineState(descriptor: descriptor)
        baseline = try device.makeComputePipelineState(function: library.makeFunction(name: "baseline")!)
        super.init()
    }

    public func attach(_ view: MTKView) {
        self.view = view; view.device = device; view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = true; view.isPaused = true; view.enableSetNeedsDisplay = false
        view.autoResizeDrawable = false; view.colorspace = CGColorSpace(name: CGColorSpace.itur_709)
        view.layer?.isOpaque = true
        cover.backgroundColor = NSColor.black.cgColor; cover.frame = view.bounds
        cover.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]; cover.actions = ["hidden": NSNull(), "bounds": NSNull(), "position": NSNull()]
        view.layer?.addSublayer(cover)
        resizeDrawable()
        startScheduler()
    }

    public func resizeDrawable() {
        guard let view = view, let layer = view.layer as? CAMetalLayer else { return }
        view.window?.contentView?.layoutSubtreeIfNeeded()
        let backing = view.convertToBacking(view.bounds).size
        guard backing.width > 0 && backing.height > 0 else { return }
        let size = CGSize(width: backing.width.rounded(), height: backing.height.rounded())
        layer.contentsScale = view.window?.backingScaleFactor ?? 1
        if view.drawableSize != size || layer.drawableSize != size {
            view.drawableSize = size; layer.drawableSize = size
            event("drawable-resize", ["drawable": [Int(size.width), Int(size.height)],
                "viewPoints": [Double(view.bounds.width), Double(view.bounds.height)], "backingScale": layer.contentsScale])
        }
        redrawStill()
    }

    private func startScheduler() {
        guard link == nil, let layer = view?.layer as? CAMetalLayer else { return }
        let link = CAMetalDisplayLink(metalLayer: layer)
        link.delegate = self; link.preferredFrameRateRange = CAFrameRateRange(minimum: 60, maximum: 60, preferred: 60)
        link.preferredFrameLatency = 1; link.add(to: .main, forMode: .common); self.link = link
    }

    public func resources() -> [String: Any] {
        ["pixelBuffers": slots.filter { $0.frame != nil }.count, "textureWrappers": slots.reduce(0) { $0 + ($1.lease?.wrappers.count ?? 0) },
         "processingSlots": slots.filter(\.processing).count, "presentationCommands": slots.filter(\.presenting).count,
         "occupiedSlots": state.active.count, "activeOutputs": source.output == nil ? 0 : 1, "displayLinks": link == nil ? 0 : 1,
         "textureCaches": bridge == nil ? 0 : 1, "configuredSlots": slots.count,
         "retainedOwners": VideoResources.snapshot()]
    }

    private func event(_ name: String, _ fields: [String: Any] = [:]) {
        onEvent?(["kind": name, "host": CACurrentMediaTime(), "generation": state.generation].merging(fields) { _, new in new })
    }

    private func invalidate(_ reason: String, playing: Bool, hide: Bool) {
        state.invalidate(playing: playing); lastConfiguration = nil; previousPTS = nil
        refresh = false
        for slot in slots { slot.current = false; slot.ready = false; releaseIfIdle(slot) }
        if hide { cover.isHidden = false }
        event("invalidate", ["reason": reason]); onStatus?()
    }

    private func releaseIfIdle(_ slot: FrameSlot) {
        if !slot.processing && !slot.presenting && !slot.current && !slot.ready {
            if let identity = slot.identity { state.release(identity) }
            slot.identity = nil; slot.frame = nil; slot.lease = nil; slot.item = nil
        }
        if slots.allSatisfy({ !$0.processing && !$0.presenting }) {
            let waiters = drainWaiters; drainWaiters.removeAll(); waiters.forEach { $0.resume() }
        }
    }

    public func open(_ url: URL) async throws {
        requestID &+= 1; let request = requestID
        await halt()
        guard request == requestID else { throw CancellationError() }
        guard !disposed && !state.terminal else { throw VideoError("Terminal GPU context requires reopening the application") }
        do { try await source.open(url) }
        catch { if request != requestID { throw CancellationError() }; throw error }
        guard request == requestID else { throw CancellationError() }
        invalidate("source", playing: false, hide: true); currentItem = source.player.currentItem.map(ObjectIdentifier.init)
        refresh = true; startScheduler(); link?.isPaused = false
        event("source", ["file": url.lastPathComponent, "fps": source.sourceFPS, "duration": source.duration, "hasAudio": source.hasAudio,
                         "shaderSHA": slots[0].network.shaderSha256, "weightHashes": slots[0].network.packedWeightHashes])
    }

    public func play() {
        guard source.url != nil && !state.terminal else { return }
        desiredPlaying = true
        if seeking || seekTarget != nil { pendingSeekResume = true; return }
        invalidate("resume", playing: true, hide: true); source.player.play(); refresh = false; startScheduler(); link?.isPaused = false
        event("play")
    }

    public func pause() {
        desiredPlaying = false; pendingSeekResume = false
        if isSeeking { event("pause-during-seek"); return }
        source.player.pause()
        state.invalidate(playing: false); link?.isPaused = true
        for slot in slots where !slot.current { slot.ready = false; releaseIfIdle(slot) }
        event("pause"); onStatus?()
    }

    public func seek(_ seconds: Double) async {
        guard source.url != nil && !state.terminal else { return }
        requestID &+= 1; let request = requestID
        pendingSeekResume = desiredPlaying
        source.player.pause(); seeking = true; invalidate("seek", playing: false, hide: true)
        link?.isPaused = false
        let target = min(max(0, seconds), max(0, source.duration - 0.1)); seekTarget = target
        source.player.currentItem?.cancelPendingSeeks()
        let success: Bool = await withCheckedContinuation { continuation in
            let completion = SeekCompletion { continuation.resume(returning: $0) }
            source.player.seek(to: CMTime(seconds: target, preferredTimescale: 60000), toleranceBefore: .zero, toleranceAfter: .zero) { success in
                Task { @MainActor in completion.finish(success) }
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 5) { completion.finish(false) }
        }
        guard request == requestID else { return }
        event("seek-completion", ["target": target, "success": success])
        guard success else { seeking = false; seekTarget = nil; fail("AVPlayer seek failed"); return }
        source.attachOutput(); seeking = false; state.invalidate(playing: false); previousPTS = nil; lastConfiguration = nil
        currentItem = source.player.currentItem.map(ObjectIdentifier.init)
        refresh = true; link?.isPaused = false
        onStatus?()
    }

    public func setNeural(_ value: Bool) {
        neural = value; let playing = state.playing; invalidate("mode", playing: playing, hide: true)
        refresh = true; link?.isPaused = false; event("mode", ["neural": value])
    }

    public func stop() async {
        requestID &+= 1
        await halt()
    }

    private func halt() async {
        seeking = false; seekTarget = nil; pendingSeekResume = false; desiredPlaying = false; currentItem = nil
        source.stop(); invalidate("stop", playing: false, hide: true); link?.invalidate(); link = nil
        if slots.contains(where: { $0.processing || $0.presenting }) { await withCheckedContinuation { drainWaiters.append($0) } }
        for slot in slots { slot.ready = false; slot.current = false; releaseIfIdle(slot) }
        bridge?.flushAfterDrain(); event("cleanup", resources())
    }

    public func dispose() async {
        await stop(); disposed = true; slots.removeAll(); bridge = nil; event("disposed", resources())
    }

    public func redrawStill() { if !state.playing && !seeking { link?.isPaused = false } }

    private func promoteReady() {
        let ready = slots.filter { $0.ready && $0.identity.map(state.accepts) == true && $0.item === source.player.currentItem }
            .max { ($0.identity?.sequence ?? 0) < ($1.identity?.sequence ?? 0) }
        if let ready = ready {
            for slot in slots where slot !== ready && (slot.current || slot.ready) { slot.current = false; slot.ready = false; releaseIfIdle(slot) }
            ready.ready = false; ready.current = true
        }
    }

    public func fail(_ message: String) {
        guard error == nil else { return }
        error = message; source.player.pause(); state.fail(); invalidate("terminal", playing: false, hide: true)
        link?.isPaused = true; event("error", ["error": message]); onStatus?()
    }

    nonisolated public func metalDisplayLink(_ link: CAMetalDisplayLink, needsUpdate update: CAMetalDisplayLink.Update) {
        MainActor.assumeIsolated { autoreleasepool { tick(update) } }
    }

    private func tick(_ update: CAMetalDisplayLink.Update) {
        opportunities &+= 1
        let opportunityHost = CACurrentMediaTime()
        if opportunityHost - lastOpportunityHost >= 1 {
            lastOpportunityHost = opportunityHost; event("heartbeat", diagnosticSnapshot())
        }
        guard !state.terminal else { return }
        if seeking {
            do { _ = try source.sample(hostTime: update.targetPresentationTimestamp) }
            catch { fail(String(describing: error)) }
            return
        }
        let item = source.player.currentItem.map(ObjectIdentifier.init)
        if item != currentItem {
            if currentItem != nil { loops &+= 1 }
            currentItem = item; invalidate("item-transition", playing: desiredPlaying, hide: true)
            event("loop", ["loop": loops, "item": item.map(String.init(describing:)) ?? "none"])
        }
        promoteReady()
        if source.player.currentItem?.status == .failed { fail(String(describing: source.player.currentItem?.error)); return }
        do {
            if state.playing || refresh, let frame = try source.sample(hostTime: update.targetPresentationTimestamp) {
                acquired &+= 1
                guard frame.item === source.player.currentItem else { duplicate &+= 1; return }
                if let last = lastConfiguration, last != frame.configuration || previousPTS.map({ frame.pts + 0.000001 < $0 }) == true {
                    invalidate("sample-discontinuity", playing: desiredPlaying, hide: true); event("discontinuity", ["pts": frame.pts])
                }
                lastConfiguration = frame.configuration
                if let target = seekTarget {
                    guard abs(frame.pts - target) <= 2 / max(1, source.sourceFPS) else { throw VideoError("Post-seek sample outside paused target") }
                }
                if previousPTS.map({ abs(frame.pts - $0) < 0.000001 }) == true { duplicate &+= 1 }
                else if let slot = slots.first(where: { $0.identity == nil }), let identity = state.acquire(pts: frame.pts, pausedDiagnostic: refresh) {
                    if let previous = previousPTS {
                        let count = Int(((frame.pts - previous) * source.sourceFPS).rounded()) - 1
                        if count > 0 { gaps &+= UInt64(count) }
                    }
                    previousPTS = frame.pts; refresh = false
                    try process(frame, identity: identity, slot: slot)
                    if let target = seekTarget {
                        event("post-seek", ["target": target, "pts": frame.pts]); seekTarget = nil
                        state.setPlaying(pendingSeekResume); if pendingSeekResume { source.player.play() }
                    }
                } else { busy &+= 1 }
            }
            if let slot = slots.first(where: { $0.current && !$0.presenting }), let identity = slot.identity,
               slot.item === source.player.currentItem && (state.accepts(identity) || !state.playing) {
                try present(slot, update: update)
            }
            onStatus?()
        } catch { fail(String(describing: error)) }
    }

    private func process(_ frame: DecodedFrame, identity: FrameIdentity, slot: FrameSlot) throws {
        slot.identity = identity
        var committed = false
        defer { if !committed { slot.processing = false; slot.frame = nil; slot.lease = nil; releaseIfIdle(slot) } }
        guard let bridge = bridge else { throw VideoError("Disposed bridge") }
        let lease = try bridge.map(frame.pixelBuffer)
        slot.item = frame.item; slot.frame = frame; slot.lease = lease; slot.recorded = false; slot.invalidOutputFlags = nil
        slot.acquisitionHost = CACurrentMediaTime()
        guard let ingest = queue.makeCommandBuffer(), let network = queue.makeCommandBuffer() else { throw VideoError("Frame command allocation failed") }
        ingest.label = "ingest"; network.label = neural ? "neural-F" : "baseline"
        try bridge.encode(lease, into: slot.network.input, command: ingest)
        if neural { try slot.network.encode(command: network) }
        else {
            guard let encoder = network.makeComputeCommandEncoder() else { throw VideoError("Baseline encoder unavailable") }
            encoder.setComputePipelineState(baseline); encoder.setBuffer(slot.network.input, offset: 0, index: 0); encoder.setTexture(slot.network.output, index: 0)
            encoder.dispatchThreads(MTLSize(width: 2560, height: 1440, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1)); encoder.endEncoding()
        }
        let mode = neural, acquiredColor = lease.color
        network.addCompletedHandler { [frame, lease, ingest, network] _ in
            withExtendedLifetime((frame, lease)) {
                let finished = CACurrentMediaTime()
                _ = Task<Void, Never> { @MainActor in
                    slot.processing = false; slot.frame = nil; slot.lease = nil; slot.completionHost = finished
                    if network.status != .completed || ingest.status != .completed || network.error != nil || ingest.error != nil {
                        self.fail("GPU frame failure: \(String(describing: network.error ?? ingest.error))")
                    } else if mode && slot.network.validationFlagsAfterCompletion() != 0 {
                        self.fail("Invalid neural output: \(slot.network.validationFlagsAfterCompletion())")
                    } else if self.state.accepts(identity) && frame.item === self.source.player.currentItem {
                        self.completed &+= 1; slot.ready = true
                        slot.invalidOutputFlags = mode ? slot.network.validationFlagsAfterCompletion() : 0
                        slot.ingestMS = Self.duration(ingest); slot.neuralMS = Self.duration(network)
                        slot.ingestStart = ingest.gpuStartTime; slot.ingestEnd = ingest.gpuEndTime
                        slot.networkStart = network.gpuStartTime; slot.networkEnd = network.gpuEndTime
                        slot.processingMS = network.gpuEndTime > ingest.gpuStartTime && ingest.gpuStartTime > 0 ? (network.gpuEndTime - ingest.gpuStartTime) * 1000 : nil
                        if self.completed == 1 || self.completed % 300 == 0 {
                            self.event("format", ["format": acquiredColor.format, "matrix": acquiredColor.matrix ?? "missing", "primaries": acquiredColor.primaries ?? "missing",
                                "transfer": acquiredColor.transfer ?? "missing", "chroma": acquiredColor.chroma ?? "missing", "planes": lease.planes.map { [$0.width, $0.height] },
                                "flipped": lease.flippedFlags, "neural": mode])
                        }
                    }
                    self.releaseIfIdle(slot)
                }
            }
        }
        submitted &+= 1; slot.processing = true; committed = true; slot.submissionHost = CACurrentMediaTime(); ingest.commit(); network.commit()
    }

    private static func duration(_ command: MTLCommandBuffer) -> Double? {
        let start = command.gpuStartTime, end = command.gpuEndTime
        return start.isFinite && start > 0 && end.isFinite && end > start ? (end - start) * 1000 : nil
    }

    private func present(_ slot: FrameSlot, update: CAMetalDisplayLink.Update) throws {
          guard let identity = slot.identity, let item = slot.item, item === source.player.currentItem,
              let command = queue.makeCommandBuffer() else { throw VideoError("Presentation command unavailable") }
        let drawable = update.drawable, descriptor = MTLRenderPassDescriptor()
        descriptor.colorAttachments[0].texture = drawable.texture; descriptor.colorAttachments[0].loadAction = .clear
        descriptor.colorAttachments[0].storeAction = .store; descriptor.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1)
        guard let encoder = command.makeRenderCommandEncoder(descriptor: descriptor) else { throw VideoError("Presentation encoder unavailable") }
        let width = Double(drawable.texture.width), height = Double(drawable.texture.height), scale = min(width / 2560, height / 1440)
        encoder.setViewport(MTLViewport(originX: (width - 2560 * scale) / 2, originY: (height - 1440 * scale) / 2,
            width: 2560 * scale, height: 1440 * scale, znear: 0, zfar: 1))
        encoder.setRenderPipelineState(render); encoder.setFragmentTexture(slot.network.output, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3); encoder.endEncoding()
        let opportunity = CACurrentMediaTime(), playerTime = item.currentTime().seconds, mode = neural
        slot.presenting = true
        command.addCompletedHandler { [drawable, command] _ in
            withExtendedLifetime(drawable) {
                _ = Task<Void, Never> { @MainActor in
                    slot.presenting = false
                    if command.status != .completed || command.error != nil { self.fail("Presentation GPU failure") }
                    else if self.state.accepts(identity) && item === self.source.player.currentItem && !slot.recorded {
                        slot.recorded = true; self.presented &+= 1; self.cover.isHidden = true
                        self.lastPresentedDrawable = CGSize(width: width, height: height)
                        self.lastPresentedOutput = CGSize(width: slot.network.output.width, height: slot.network.output.height)
                        let copy = Self.duration(command)
                        let total = slot.ingestMS.flatMap { ingest in slot.neuralMS.flatMap { neural in copy.map { ingest + neural + $0 } } }
                        self.event("frame", ["sequence": identity.sequence, "frameGeneration": identity.generation, "pts": identity.presentationTime,
                            "playerTime": playerTime, "ageMS": (playerTime - identity.presentationTime) * 1000,
                            "itemMatched": item === self.source.player.currentItem,
                            "itemIdentity": String(describing: ObjectIdentifier(item)), "sourceFile": self.source.url?.lastPathComponent ?? "none",
                            "invalidOutputFlags": slot.invalidOutputFlags as Any? ?? NSNull(),
                            "outputValidation": mode ? "hidden-final-gpu" : "format-validated-unorm-baseline",
                            "gpuIngestStart": slot.ingestStart, "gpuIngestEnd": slot.ingestEnd,
                            "gpuNetworkStart": slot.networkStart, "gpuNetworkEnd": slot.networkEnd,
                            "gpuPresentationStart": command.gpuStartTime, "gpuPresentationEnd": command.gpuEndTime,
                            "acquisitionHost": slot.acquisitionHost, "submissionHost": slot.submissionHost, "completionHost": slot.completionHost,
                            "opportunityHost": opportunity, "presentationCompleteHost": CACurrentMediaTime(), "neural": mode,
                            "gpuIngestMS": slot.ingestMS as Any? ?? NSNull(), "gpuNeuralMS": slot.neuralMS as Any? ?? NSNull(),
                            "gpuProcessingSpanMS": slot.processingMS as Any? ?? NSNull(), "gpuPresentationMS": copy as Any? ?? NSNull(), "gpuFramePathMS": total as Any? ?? NSNull(),
                            "rate": self.source.player.rate, "timeControl": self.source.player.timeControlStatus.rawValue,
                            "sourceFPS": self.source.sourceFPS, "drawable": [Int(width), Int(height)],
                            "networkOutput": [slot.network.output.width, slot.network.output.height], "resources": self.resources(),
                            "acquired": self.acquired, "submitted": self.submitted, "completed": self.completed, "presented": self.presented,
                            "busy": self.busy, "duplicates": self.duplicate, "gaps": self.gaps, "loops": self.loops])
                    }
                    if self.state.accepts(identity) && !self.state.playing && !self.isSeeking && !self.refresh &&
                        !self.slots.contains(where: { $0.processing || $0.ready }) { self.link?.isPaused = true }
                    self.releaseIfIdle(slot)
                }
            }
        }
        command.present(drawable); command.commit()
    }
}