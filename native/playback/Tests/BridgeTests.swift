import CoreVideo
import Foundation
import Metal
import XCTest
@testable import NativeVideo

final class BridgeTests: XCTestCase {
    func testFormatMetadataRejection() throws {
        let buffer = try fixture(format: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
        XCTAssertNoThrow(try VideoColor(buffer))
        CVBufferSetAttachment(buffer, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_601_4, .shouldPropagate)
        XCTAssertThrowsError(try VideoColor(buffer))
        CVBufferRemoveAllAttachments(buffer); XCTAssertThrowsError(try VideoColor(buffer))
    }

    func testMalformedColorGeometryAndFields() throws {
        let mutations: [(CFString, CFTypeRef?)] = [
            (kCVImageBufferChromaLocationTopFieldKey, nil), (kCVImageBufferFieldCountKey, 2 as CFNumber),
            (kCVImageBufferChromaLocationBottomFieldKey, kCVImageBufferChromaLocation_Center),
            (kCVImageBufferCleanApertureKey, "invalid" as CFString),
            (kCVImageBufferCleanApertureKey, [kCVImageBufferCleanApertureWidthKey: 10.5, kCVImageBufferCleanApertureHeightKey: 6,
                kCVImageBufferCleanApertureHorizontalOffsetKey: 0, kCVImageBufferCleanApertureVerticalOffsetKey: 0] as CFDictionary),
            (kCVImageBufferPixelAspectRatioKey, [:] as CFDictionary),
            (kCVImageBufferPixelAspectRatioKey, [kCVImageBufferPixelAspectRatioHorizontalSpacingKey: 0, kCVImageBufferPixelAspectRatioVerticalSpacingKey: 0] as CFDictionary)]
        for (key, value) in mutations {
            let buffer = try fixture(format: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, metal: false)
            if let value = value { CVBufferSetAttachment(buffer, key, value, .shouldPropagate) } else { CVBufferRemoveAttachment(buffer, key) }
            if value != nil { XCTAssertNotNil(CVBufferCopyAttachment(buffer, key, nil)) }
            XCTAssertThrowsError(try VideoColor(buffer), "\(key)")
        }
    }

    func testLeaseReleasedAfterCompletion() throws {
        guard ProcessInfo.processInfo.environment["AETHERVSR_METAL_TESTS"] == "1" else { throw XCTSkip("Physical bridge lifetime test is opt-in") }
        weak var released: VideoLease?
        try autoreleasepool {
            let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), bridge = try VideoBridge(device: device)
            var lease: VideoLease? = try bridge.map(fixture(format: kCVPixelFormatType_32BGRA))
            released = lease
            var command: MTLCommandBuffer? = device.makeCommandQueue()!.makeCommandBuffer()
            let output = device.makeBuffer(length: 10 * 6 * 16, options: .storageModeShared)!
            try bridge.encode(lease!, into: output, command: command!)
            command!.addCompletedHandler { [owner = lease!] _ in withExtendedLifetime(owner) {} }
            lease = nil; XCTAssertNotNil(released)
            command!.commit(); command!.waitUntilCompleted(); command = nil
        }
        XCTAssertNil(released)
    }

    func testMappedContentAndColor() throws {
        guard ProcessInfo.processInfo.environment["AETHERVSR_METAL_TESTS"] == "1" else { throw XCTSkip("Physical Core Video/Metal test is opt-in") }
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), bridge = try VideoBridge(device: device)
        for format in [kCVPixelFormatType_32BGRA, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange] {
            for centered in [false, true] {
                var pixelBuffer: CVPixelBuffer? = try fixture(format: format, centered: centered)
                var lease: VideoLease? = try bridge.map(pixelBuffer!)
                let expected = cpuReference(pixelBuffer!, centered: centered)
                let width = CVPixelBufferGetWidth(pixelBuffer!), height = CVPixelBufferGetHeight(pixelBuffer!)
                XCTAssertEqual(lease!.planes.count, format == kCVPixelFormatType_32BGRA ? 1 : 2)
                let output = try XCTUnwrap(device.makeBuffer(length: width * height * 16, options: .storageModeShared))
                let command = try XCTUnwrap(device.makeCommandQueue()!.makeCommandBuffer())
                try bridge.encode(lease!, into: output, command: command)
                weak var weakLease = lease
                command.addCompletedHandler { [retained = lease!] _ in withExtendedLifetime(retained) {} }
                pixelBuffer = nil; lease = nil
                XCTAssertNotNil(weakLease)
                command.commit(); command.waitUntilCompleted()
                XCTAssertEqual(command.status, .completed); XCTAssertNil(command.error)
                let actual = output.contents().assumingMemoryBound(to: Float.self)
                for index in expected.indices { XCTAssertTrue(actual[index].isFinite); XCTAssertEqual(Double(actual[index]), expected[index], accuracy: 0.00001, "format\(format) index\(index)") }
            }
        }
        bridge.flushAfterDrain()
    }
}

func fixture(format: OSType, centered: Bool = false, metal: Bool = true) throws -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    let width = 10, height = 6
    let attributes: [String: Any] = [kCVPixelBufferMetalCompatibilityKey as String: true,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:], kCVPixelBufferBytesPerRowAlignmentKey as String: 64]
    guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, metal ? attributes as CFDictionary : nil, &buffer) == kCVReturnSuccess,
          let buffer = buffer else { throw VideoError("Fixture allocation failed") }
    CVBufferSetAttachment(buffer, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
    CVBufferSetAttachment(buffer, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
    CVBufferSetAttachment(buffer, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
    CVBufferSetAttachment(buffer, kCVImageBufferChromaLocationTopFieldKey, centered ? kCVImageBufferChromaLocation_Center : kCVImageBufferChromaLocation_Left, .shouldPropagate)
    CVPixelBufferLockBaseAddress(buffer, []); defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let codes: [UInt8] = [0, 16, 128, 235, 240, 255, 81, 145, 41, 210]
    if format == kCVPixelFormatType_32BGRA {
        let pointer = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self), stride = CVPixelBufferGetBytesPerRow(buffer)
        for row in 0..<height { for column in 0..<width {
            for channel in 0..<3 { pointer[row * stride + column * 4 + channel] = codes[(column + row * 3 + channel * 2) % codes.count] }
            pointer[row * stride + column * 4 + 3] = 255
        } }
    } else {
        let luma = CVPixelBufferGetBaseAddressOfPlane(buffer, 0)!.assumingMemoryBound(to: UInt8.self)
        let chroma = CVPixelBufferGetBaseAddressOfPlane(buffer, 1)!.assumingMemoryBound(to: UInt8.self)
        for row in 0..<height { for column in 0..<width { luma[row * CVPixelBufferGetBytesPerRowOfPlane(buffer, 0) + column] = codes[(column + row * 3) % codes.count] } }
        for row in 0..<(height / 2) { for column in 0..<(width / 2) {
            let index = row * CVPixelBufferGetBytesPerRowOfPlane(buffer, 1) + column * 2
            chroma[index] = codes[(column + row * 2) % codes.count]; chroma[index + 1] = codes[(column * 3 + row + 2) % codes.count]
        } }
    }
    return buffer
}

func cpuReference(_ buffer: CVPixelBuffer, centered: Bool) -> [Double] {
    CVPixelBufferLockBaseAddress(buffer, .readOnly); defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer), format = CVPixelBufferGetPixelFormatType(buffer)
    var result = [Double](repeating: 0, count: width * height * 4)
    for row in 0..<height { for column in 0..<width {
        var rgb: [Double]
        if format == kCVPixelFormatType_32BGRA {
            let pointer = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self), index = row * CVPixelBufferGetBytesPerRow(buffer) + column * 4
            rgb = [Double(pointer[index + 2]), Double(pointer[index + 1]), Double(pointer[index])].map { $0 / 255 }
        } else {
            let luma = CVPixelBufferGetBaseAddressOfPlane(buffer, 0)!.assumingMemoryBound(to: UInt8.self)
            let uv = CVPixelBufferGetBaseAddressOfPlane(buffer, 1)!.assumingMemoryBound(to: UInt8.self)
            let horizontal = Double(column) / 2 - (centered ? 0.25 : 0), vertical = Double(row) / 2 - 0.25
            func sample(_ channel: Int) -> Double {
                let left = Int(floor(horizontal)), top = Int(floor(vertical)), alpha = horizontal - floor(horizontal), beta = vertical - floor(vertical)
                func code(_ column: Int, _ row: Int) -> Double {
                    Double(uv[min(height / 2 - 1, max(0, row)) * CVPixelBufferGetBytesPerRowOfPlane(buffer, 1) + min(width / 2 - 1, max(0, column)) * 2 + channel])
                }
                return (code(left, top) * (1 - alpha) + code(left + 1, top) * alpha) * (1 - beta) +
                    (code(left, top + 1) * (1 - alpha) + code(left + 1, top + 1) * alpha) * beta
            }
            let video = format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
            let luminance = (Double(luma[row * CVPixelBufferGetBytesPerRowOfPlane(buffer, 0) + column]) - (video ? 16 : 0)) / (video ? 219 : 255)
            let blueDifference = (sample(0) - 128) / (video ? 224 : 255), redDifference = (sample(1) - 128) / (video ? 224 : 255)
            let red = luminance + 2 * (1 - 0.2126) * redDifference
            let blue = luminance + 2 * (1 - 0.0722) * blueDifference
            let green = (luminance - 0.2126 * red - 0.0722 * blue) / 0.7152
            rgb = [red, green, blue]
        }
        for channel in 0..<3 { result[(row * width + column) * 4 + channel] = min(1, max(0, rgb[channel])) }
    } }
    return result
}