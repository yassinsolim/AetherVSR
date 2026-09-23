// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AetherNativePlayback",
    platforms: [.macOS("26.0")],
    products: [.library(name: "NativeVideo", targets: ["NativeVideo"]), .library(name: "PlaybackCore", targets: ["PlaybackCore"]),
        .executable(name: "aether-player", targets: ["NativePlayer"])],
    targets: [
        .target(name: "AetherMetal", path: ".",
            exclude: ["macos/Package.swift", "macos/Tests", "macos/Sources/MetalQualification", "macos/Sources/OptimizedQualification", "playback/Bridge", "playback/Tests", "playback/Player", "playback/App", "playback/Diagnostics"],
            sources: ["macos/Sources/AetherMetal/Model.swift", "macos/Sources/AetherMetal/Validation.swift",
                "macos/Sources/AetherMetal/MetalEngine.swift", "macos/Sources/AetherMetal/PackedLayout.swift",
                "macos/Sources/AetherMetal/OptimizedShaders.swift", "macos/Sources/AetherMetal/OptimizedEngine.swift", "playback/Engine"],
            resources: [.copy("macos/Sources/AetherMetal/Shaders.metal")]),
        .target(name: "NativeVideo", path: "playback/Bridge", resources: [.copy("Ingest.metal")]),
        .target(name: "PlaybackCore", dependencies: ["NativeVideo", "AetherMetal"], path: "playback/Player", resources: [.copy("Present.metal")]),
        .target(name: "PlaybackDiagnostics", dependencies: ["PlaybackCore", "NativeVideo", "AetherMetal"], path: "playback/Diagnostics"),
        .executableTarget(name: "NativePlayer", dependencies: ["PlaybackCore", "NativeVideo", "AetherMetal", "PlaybackDiagnostics"], path: "playback/App"),
        .testTarget(name: "NativeVideoTests", dependencies: ["NativeVideo", "AetherMetal", "PlaybackCore"], path: "playback/Tests")
    ],
    swiftLanguageModes: [.v5]
)