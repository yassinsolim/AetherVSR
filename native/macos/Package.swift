// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "AetherMetal",
    platforms: [.macOS(.v13)],
    products: [.library(name: "AetherMetal", targets: ["AetherMetal"]),
               .executable(name: "aether-metal", targets: ["MetalQualification"]),
               .executable(name: "aether-metal-opt", targets: ["OptimizedQualification"])],
    targets: [
        .target(name: "AetherMetal", resources: [.copy("Shaders.metal")]),
        .executableTarget(name: "MetalQualification", dependencies: ["AetherMetal"]),
        .executableTarget(name: "OptimizedQualification", dependencies: ["AetherMetal"]),
        .testTarget(name: "AetherMetalTests", dependencies: ["AetherMetal"])
    ]
)