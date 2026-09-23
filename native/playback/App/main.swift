import AetherMetal
import AppKit
import Foundation
import MetalKit
import PlaybackCore
import PlaybackDiagnostics
import QuartzCore
import UniformTypeIdentifiers

@MainActor final class PlayerApp: NSObject, NSApplicationDelegate, NSWindowDelegate {
    var window: NSWindow!
    var controller: PlaybackController!
    let timeline = NSSlider(value: 0, minValue: 0, maxValue: 1, target: nil, action: nil)
    let status = NSTextField(labelWithString: "No video")
    let playButton = NSButton()
    var log: FileHandle?
    var diagnosticOutput: URL?
    var diagnosticStart: Double?
    var closing = false
    var model: ProductionModel!
    var diagnosticMode = ""
    var observationSeconds = 0.0
    var measurementStart: Double?
    var measurementFinished = false
    var diagnosticFailure: String?
    var checks: [[String: Any]] = []
    var waiters: [UUID: (() -> Bool, CheckedContinuation<Void, Error>)] = [:]

    func applicationDidFinishLaunching(_ notification: Notification) {
        do {
            model = try ProductionModel(data: Data(contentsOf: URL(fileURLWithPath: "public/models/aethersr-c16d2.json")))
            guard let device = MTLCreateSystemDefaultDevice() else { throw QualificationError("Metal unavailable") }
            controller = try PlaybackController(model: model, device: device)
            window = NSWindow(contentRect: NSRect(x: 100, y: 100, width: 1000, height: 640), styleMask: [.titled, .closable, .resizable, .miniaturizable], backing: .buffered, defer: false)
            window.title = "AetherVSR Native"; window.delegate = self; window.minSize = NSSize(width: 560, height: 360)
            window.collectionBehavior = [.fullScreenPrimary]
            let video = MTKView(frame: .zero, device: device)
            let open = button("folder", "Open Video", #selector(openVideo))
            playButton.image = NSImage(systemSymbolName: "play.fill", accessibilityDescription: "Play/Pause"); playButton.target = self; playButton.action = #selector(playPause); playButton.toolTip = "Play/Pause"
            let mute = button("speaker.slash", "Mute", #selector(toggleMute))
            let volume = NSSlider(value: 1, minValue: 0, maxValue: 1, target: self, action: #selector(volumeChanged(_:))); volume.toolTip = "Volume"
            let modes = NSSegmentedControl(labels: ["Baseline", "Neural"], trackingMode: .selectOne, target: self, action: #selector(modeChanged(_:))); modes.selectedSegment = 1
            let fullscreen = button("arrow.up.left.and.arrow.down.right", "Fullscreen", #selector(fullscreen))
            timeline.target = self; timeline.action = #selector(seekChanged); timeline.isContinuous = false; timeline.toolTip = "Timeline"
            let controls = NSStackView(views: [open, playButton, timeline, mute, volume, modes, fullscreen]); controls.spacing = 10; controls.orientation = .horizontal
            let stack = NSStackView(views: [video, controls, status]); stack.orientation = .vertical; stack.spacing = 8
            stack.edgeInsets = NSEdgeInsets(top: 8, left: 10, bottom: 8, right: 10); stack.translatesAutoresizingMaskIntoConstraints = false
            let content = window.contentView!; content.addSubview(stack)
            NSLayoutConstraint.activate([stack.leadingAnchor.constraint(equalTo: content.leadingAnchor), stack.trailingAnchor.constraint(equalTo: content.trailingAnchor),
                stack.topAnchor.constraint(equalTo: content.topAnchor), stack.bottomAnchor.constraint(equalTo: content.bottomAnchor),
                video.widthAnchor.constraint(equalTo: stack.widthAnchor, constant: -20), video.heightAnchor.constraint(greaterThanOrEqualToConstant: 240),
                controls.widthAnchor.constraint(equalTo: video.widthAnchor), timeline.widthAnchor.constraint(greaterThanOrEqualToConstant: 120), volume.widthAnchor.constraint(equalToConstant: 80)])
            controller.onStatus = { [weak self] in self?.updateStatus() }
            controller.onEvent = { [weak self] value in self?.record(value) }
            window.makeKeyAndOrderFront(nil); NSApp.activate(); controller.attach(video)
            let arguments = Array(CommandLine.arguments.dropFirst())
            if !arguments.isEmpty {
                let measure = arguments.first == "--measure"
                guard (measure && arguments.count == 5 && ["baseline", "neural"].contains(arguments[1])) ||
                    (arguments.count == 3 && ["--smoke", "--lifecycle", "--parity"].contains(arguments[0])) else { throw QualificationError("Use --smoke|--lifecycle|--parity CLIP NEW_OUTPUT or --measure baseline|neural CLIP NEW_OUTPUT SECONDS") }
                diagnosticMode = String(arguments[0].dropFirst(2))
                observationSeconds = measure ? Double(arguments[4]) ?? 0 : 8
                if measure && ![60.0, 600.0].contains(observationSeconds) { throw QualificationError("Unregistered observation duration") }
                let clip = arguments[measure ? 2 : 1]
                let output = URL(fileURLWithPath: arguments[measure ? 3 : 2], isDirectory: true)
                guard !FileManager.default.fileExists(atPath: output.path) else { throw QualificationError("Diagnostic output already exists") }
                try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
                diagnosticOutput = output; let path = output.appendingPathComponent("events.jsonl")
                try Data().write(to: path, options: .withoutOverwriting); log = try FileHandle(forWritingTo: path)
                record(["kind": "environment", "host": CACurrentMediaTime(), "os": ProcessInfo.processInfo.operatingSystemVersionString,
                    "device": device.name, "thermal": ProcessInfo.processInfo.thermalState.rawValue,
                    "screenMaxFPS": window.screen?.maximumFramesPerSecond ?? 0, "backingScale": window.backingScaleFactor,
                    "modelSHA": ProductionModel.fileHash, "mode": diagnosticMode])
                Task { @MainActor in
                    do {
                        try await controller.open(URL(fileURLWithPath: clip))
                        if measure { controller.setNeural(arguments[1] == "neural") }
                        controller.play()
                        if diagnosticMode == "lifecycle" { try await lifecycle(); await finish() }
                        if diagnosticMode == "parity" { try await parity(); await finish() }
                    }
                    catch is CancellationError {} catch { controller.fail(String(describing: error)); await finish() }
                }
                let timeout = measure ? observationSeconds + 40 : diagnosticMode == "parity" ? 180 : diagnosticMode == "lifecycle" ? 90 : 20
                DispatchQueue.main.asyncAfter(deadline: .now() + timeout) {
                    guard !self.closing else { return }
                    self.diagnosticFailure = "Diagnostic deadline"; Task { @MainActor in await self.finish() }
                }
            }
        } catch { FileHandle.standardError.write(Data("\(error)\n".utf8)); NSApp.terminate(nil) }
    }

    func button(_ symbol: String, _ title: String, _ action: Selector) -> NSButton {
        let button = NSButton(image: NSImage(systemSymbolName: symbol, accessibilityDescription: title)!, target: self, action: action)
        button.toolTip = title; return button
    }
    func record(_ event: [String: Any]) {
        if let log = log {
            do {
                let data = try JSONSerialization.data(withJSONObject: event, options: [.sortedKeys])
                try log.write(contentsOf: data); try log.write(contentsOf: Data([10]))
            } catch {
                diagnosticFailure = "Evidence write failed: \(error)"; self.log = nil
                Task { await finish() }; return
            }
        }
        if diagnosticOutput != nil {
            if event["kind"] as? String == "error" && diagnosticMode != "lifecycle" { Task { await finish() } }
            if event["kind"] as? String == "frame" {
                let now = event["opportunityHost"] as! Double
                if diagnosticStart == nil { diagnosticStart = now }
                if diagnosticMode == "smoke" && now - diagnosticStart! >= 8 { measurementFinished = true; Task { await finish() } }
                if diagnosticMode == "measure" {
                    if measurementStart == nil && now - diagnosticStart! >= 5 {
                        measurementStart = now
                        record(["kind": "measurement-start", "host": now, "duration": observationSeconds,
                            "thermal": ProcessInfo.processInfo.thermalState.rawValue, "neural": controller.neural])
                    } else if let start = measurementStart, now - start >= observationSeconds, !measurementFinished {
                        measurementFinished = true
                        record(["kind": "measurement-end", "host": now, "thermal": ProcessInfo.processInfo.thermalState.rawValue,
                            "resources": controller.resources(), "windowVisible": window.isVisible,
                            "windowKey": window.isKeyWindow, "occlusionVisible": window.occlusionState.contains(.visible)])
                        Task { await finish() }
                    }
                }
            }
        }
        checkWaiters()
    }
    func updateStatus() {
        timeline.maxValue = max(1, controller.source.duration)
        if !timeline.isHighlighted { timeline.doubleValue = controller.source.player.currentTime().seconds.isFinite ? controller.source.player.currentTime().seconds : 0 }
        status.stringValue = controller.error ?? (controller.source.url?.lastPathComponent ?? "No video")
        playButton.image = NSImage(systemSymbolName: controller.playbackIntended ? "pause.fill" : "play.fill", accessibilityDescription: "Play/Pause")
        checkWaiters()
    }

    func checkWaiters() {
        let ready = waiters.filter { $0.value.0() }.map(\.key)
        for id in ready { waiters.removeValue(forKey: id)?.1.resume() }
    }
    func wait(_ name: String, timeout: Double = 10, until predicate: @escaping () -> Bool) async throws {
        if predicate() { return }
        let id = UUID()
        try await withCheckedThrowingContinuation { continuation in
            waiters[id] = (predicate, continuation)
            DispatchQueue.main.asyncAfter(deadline: .now() + timeout) {
                self.waiters.removeValue(forKey: id)?.1.resume(throwing: QualificationError("Timeout: \(name)"))
            }
        }
    }
    func delay(_ seconds: Double) async {
        await withCheckedContinuation { continuation in DispatchQueue.main.asyncAfter(deadline: .now() + seconds) { continuation.resume() } }
    }
    func check(_ name: String, _ passed: Bool) throws {
        checks.append(["name": name, "passed": passed]); record(["kind": "check", "host": CACurrentMediaTime(), "name": name, "passed": passed])
        if !passed { throw QualificationError("Lifecycle failed: \(name)") }
    }
    func lifecycle() async throws {
        try await wait("initial frames") { self.controller.presented >= 30 }
        try check("720p30 initial native frames", controller.error == nil)
        controller.pause(); await delay(0.2)
        let pausedSubmitted = controller.submitted
        await delay(0.5); try check("pause stops neural submission", controller.submitted == pausedSubmitted)
        try check("pause remains AVPlayer authority", controller.source.player.rate == 0)
        var count = controller.presented; controller.play()
        try await wait("resume") { self.controller.presented >= count + 10 }; try check("resume advances unique output", true)
        count = controller.presented; await controller.seek(3)
        try await wait("forward seek") { !self.controller.isSeeking && self.controller.presented > count }
        try check("forward seek recovered", controller.error == nil)
        count = controller.presented; await controller.seek(1)
        try await wait("backward seek") { !self.controller.isSeeking && self.controller.presented > count }; try check("backward seek recovered", true)
        let seeking = Task { await controller.seek(2) }
        try await wait("seek in flight") { self.controller.isSeeking }; controller.pause(); await seeking.value
        try await wait("paused seek output") { !self.controller.isSeeking }; try check("pause during seek preserved", controller.error == nil && controller.source.player.rate == 0 && !controller.state.playing)
        controller.play(); count = controller.presented
        try await wait("resume after paused seek") { self.controller.presented > count + 5 }
        let firstSeek = Task { await controller.seek(3) }
        try await wait("first concurrent seek") { self.controller.isSeeking }
        let secondSeek = Task { await controller.seek(1) }; await firstSeek.value; await secondSeek.value
        try await wait("newest seek output") { !self.controller.isSeeking }; try check("successive seeks preserve play intent", controller.state.playing)
        controller.source.player.volume = 0.4; controller.source.player.isMuted = true
        try check("AVPlayer mute and volume", abs(controller.source.player.volume - 0.4) < 0.00001 && controller.source.player.isMuted)
        controller.source.player.isMuted = false; controller.source.player.volume = 1
        count = controller.presented; controller.setNeural(false)
        try await wait("baseline mode") { self.controller.presented > count + 5 }; try check("baseline same source path", !controller.neural && controller.error == nil)
        controller.setNeural(true); count = controller.presented
        try await wait("neural mode") { self.controller.presented > count + 5 }
        window.setContentSize(NSSize(width: 640, height: 400)); count = controller.presented
        try await wait("resize smaller") { self.controller.presented > count + 5 }
        window.setContentSize(NSSize(width: 1000, height: 640)); try check("resize preserves source/output contract", controller.source.sourceFPS > 0 && controller.error == nil)
        window.toggleFullScreen(nil)
        try await wait("fullscreen enter", timeout: 15) { self.window.styleMask.contains(.fullScreen) }
        await delay(1); count = controller.presented
        try await wait("fullscreen output") { self.controller.presented > count + 5 }; try check("fullscreen enter", true)
        window.toggleFullScreen(nil)
        try await wait("fullscreen exit", timeout: 15) { !self.window.styleMask.contains(.fullScreen) }
        await delay(1); try check("fullscreen exit", controller.error == nil)
        try await controller.open(URL(fileURLWithPath: "public/media/aethervsr-testclip-720p60-h264.mp4"))
        try check("replacement hides old output", controller.surfaceCovered)
        controller.play(); count = controller.presented
        try await wait("replacement frames") { self.controller.presented > count + 10 }
        try check("720p60 becomes sole authority", controller.source.url?.lastPathComponent == "aethervsr-testclip-720p60-h264.mp4" && controller.error == nil)
        let staleOpen = Task { try await controller.open(URL(fileURLWithPath: "public/media/aethervsr-testclip-720p30-h264.mp4")) }
        await delay(0)
        try await controller.open(URL(fileURLWithPath: "public/media/aethervsr-testclip-720p60-h264.mp4"))
        do { try await staleOpen.value } catch is CancellationError {}
        try check("superseded source cannot poison current", controller.error == nil && controller.source.url?.lastPathComponent == "aethervsr-testclip-720p60-h264.mp4")
        controller.play(); count = controller.presented
        try await wait("final source frames") { self.controller.presented > count + 5 }
        controller.fail("diagnostic-software-boundary")
        let failedCount = controller.submitted; controller.play(); await delay(0.2)
        try check("terminal enhancement hides and stops", controller.surfaceCovered && controller.submitted == failedCount && !controller.state.playing)
        measurementFinished = true
    }
    func parity() async throws {
        try await wait("initial parity readiness") { self.controller.presented > 5 }
        controller.pause()
        for seconds in [1.0, 2.0, 3.0] {
            let count = controller.presented; await controller.seek(seconds)
            try await wait("paused parity seek") { !self.controller.isSeeking && self.controller.presented > count }
            guard let frame = try controller.source.sample(hostTime: CACurrentMediaTime()) else { throw QualificationError("Missing paused decoded sample") }
            let result = try PausedParity.run(frame: frame, device: controller.device, model: model,
                directory: diagnosticOutput!.appendingPathComponent("frame-\(Int(seconds))"))
            try check("exact decoded parity \(Int(seconds))", result["outcome"] as? String == "PASS")
        }
        measurementFinished = true
    }
    @objc func openVideo() {
        controller.pause(); let panel = NSOpenPanel(); panel.allowedContentTypes = [.mpeg4Movie]; panel.allowsMultipleSelection = false
        panel.beginSheetModal(for: window) { response in
            guard response == .OK, let url = panel.url else { return }
            Task { @MainActor in do { try await self.controller.open(url) } catch is CancellationError {} catch { self.controller.fail(String(describing: error)) } }
        }
    }
    @objc func playPause() { controller.playbackIntended ? controller.pause() : controller.play() }
    @objc func seekChanged() { Task { await controller.seek(timeline.doubleValue) } }
    @objc func toggleMute() { controller.source.player.isMuted.toggle() }
    @objc func volumeChanged(_ sender: NSSlider) { controller.source.player.volume = Float(sender.doubleValue) }
    @objc func modeChanged(_ sender: NSSegmentedControl) { controller.setNeural(sender.selectedSegment == 1) }
    @objc func fullscreen() { window.toggleFullScreen(nil) }
    func windowShouldClose(_ sender: NSWindow) -> Bool { Task { await finish() }; return false }
    func windowDidResize(_ notification: Notification) { controller?.redrawStill() }
    func windowDidEnterFullScreen(_ notification: Notification) { record(["kind": "fullscreen-enter", "host": CACurrentMediaTime()]); controller.redrawStill() }
    func windowDidExitFullScreen(_ notification: Notification) { record(["kind": "fullscreen-exit", "host": CACurrentMediaTime()]); controller.redrawStill() }
    func finish() async {
        guard !closing else { return }; closing = true
        let waiting = waiters; waiters.removeAll(); waiting.values.forEach { $0.1.resume(throwing: CancellationError()) }
        await controller.dispose()
        if let output = diagnosticOutput {
            let expectedFault = diagnosticMode == "lifecycle" && controller.error == "diagnostic-software-boundary" && measurementFinished
            let result: [String: Any] = ["schema": "aethervsr.m13.phase2-observation/1", "mode": diagnosticMode, "presented": controller.presented, "completed": controller.completed,
                "submitted": controller.submitted, "acquired": controller.acquired, "error": controller.error as Any? ?? NSNull(), "cleanup": controller.resources(),
                "diagnosticFailure": diagnosticFailure as Any? ?? NSNull(), "checks": checks, "measurementFinished": measurementFinished,
                "outcome": (controller.error == nil || expectedFault) && diagnosticFailure == nil && measurementFinished && controller.presented > 0 ? "PASS" : "FAIL"]
            if let data = try? JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]) {
                try? data.write(to: output.appendingPathComponent("result.json"), options: .withoutOverwriting); print(String(decoding: data, as: UTF8.self))
            }
            try? log?.close(); log = nil
        }
        NSApp.terminate(nil)
    }
}

MainActor.assumeIsolated {
    let application = NSApplication.shared
    application.setActivationPolicy(.regular)
    let delegate = PlayerApp()
    application.delegate = delegate
    application.run()
}