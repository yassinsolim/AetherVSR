// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "AetherMetal",
    platforms: [.macOS(.v13)],
    products: [.library(name: "AetherMetal", targets: ["AetherMetal"]),
               .executable(name: "aether-metal", targets: ["MetalQualification"])],
    targets: [
        .target(name: "AetherMetal", resources: [.copy("Shaders.metal")]),
        .executableTarget(name: "MetalQualification", dependencies: ["AetherMetal"]),
        .testTarget(name: "AetherMetalTests", dependencies: ["AetherMetal"])
    ]
)