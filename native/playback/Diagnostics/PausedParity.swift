import AetherMetal
import CoreVideo
import Foundation
import Metal
import NativeVideo
import PlaybackCore

public enum PausedParity {
    public static func run(frame: DecodedFrame, device: MTLDevice, model: ProductionModel, directory: URL) throws -> [String: Any] {
        guard !FileManager.default.fileExists(atPath: directory.path) else { throw VideoError("Parity attempt already exists") }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let color = try VideoColor(frame.pixelBuffer), extent = try Extent(width: color.width, height: color.height)
        let referenceInput = reference(frame.pixelBuffer, color: color)
        var planes: [[String: Any]] = []
        CVPixelBufferLockBaseAddress(frame.pixelBuffer, .readOnly)
        for index in 0..<CVPixelBufferGetPlaneCount(frame.pixelBuffer) {
            let stride = CVPixelBufferGetBytesPerRowOfPlane(frame.pixelBuffer, index), height = CVPixelBufferGetHeightOfPlane(frame.pixelBuffer, index)
            let data = Data(bytes: CVPixelBufferGetBaseAddressOfPlane(frame.pixelBuffer, index)!, count: stride * height)
            let name = "plane-\(index).bin"; try data.write(to: directory.appendingPathComponent(name), options: .withoutOverwriting)
            planes.append(["file": name, "sha256": sha256(data), "bytes": data.count, "stride": stride, "height": height])
        }
        CVPixelBufferUnlockBaseAddress(frame.pixelBuffer, .readOnly)
        var runs: [[String: Any]] = []
        func artifact(_ name: String, _ values: [Float]) throws -> [String: Any] {
            let data = values.map { $0.bitPattern.littleEndian }.withUnsafeBytes { Data($0) }
            try data.write(to: directory.appendingPathComponent(name), options: .withoutOverwriting)
            return ["file": name, "bytes": data.count, "sha256": sha256(data)]
        }
        for precision in Precision.allCases {
            try autoreleasepool {
                let bridge = try VideoBridge(device: device), lease = try bridge.map(frame.pixelBuffer)
                let integrated = try FrameNetwork(device: device, model: model, extent: extent, precision: precision, diagnostic: true)
                let queue = device.makeCommandQueue()!, command = queue.makeCommandBuffer()!
                try bridge.encode(lease, into: integrated.input, command: command)
                try integrated.encode(command: command); command.commit(); command.waitUntilCompleted()
                guard command.status == .completed && command.error == nil else { throw VideoError("Parity GPU execution failed") }
                guard integrated.validationFlagsAfterCompletion() == 0 else { throw VideoError("Invalid parity neural output") }
                let pointer = integrated.input.contents().assumingMemoryBound(to: Float.self)
                var inputMax = 0.0, inputSum = 0.0, inputFailures = 0, planar = [Float](repeating: 0, count: extent.pixels * 3)
                for pixel in 0..<extent.pixels { for channel in 0..<3 {
                    let index = pixel * 4 + channel, difference = abs(Double(pointer[index]) - referenceInput[index])
                    inputMax = max(inputMax, difference); inputSum += difference
                    if !difference.isFinite || difference > 0.00001 { inputFailures += 1 }
                    planar[channel * extent.pixels + pixel] = Float(referenceInput[index])
                } }
                let original = try OptimizedEngine(model: model, extent: extent, precision: precision, candidate: .f, diagnostic: true)
                defer { original.destroy() }
                try original.loadInput(planar); let expected = try original.capture(), actual = try integrated.diagnosticStagesAfterCompletion()
                let comparisons = try Validation.qualify(actual, expected: expected.checkpoints, precision: precision, declaredPrecision: precision.rawValue)
                let stride = ((extent.width * 8 + 255) / 256) * 256
                let readback = device.makeBuffer(length: stride * extent.height * 2, options: .storageModeShared)!, copy = queue.makeCommandBuffer()!, blit = copy.makeBlitCommandEncoder()!
                blit.copy(from: integrated.output, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                    sourceSize: MTLSize(width: extent.width * 2, height: extent.height * 2, depth: 1), to: readback, destinationOffset: 0,
                    destinationBytesPerRow: stride, destinationBytesPerImage: stride * extent.height * 2); blit.endEncoding(); copy.commit(); copy.waitUntilCompleted()
                guard copy.status == .completed && copy.error == nil else { throw VideoError("Parity readback failed") }
                var rgba = Data()
                for row in 0..<(extent.height * 2) { rgba.append(readback.contents().advanced(by: row * stride).assumingMemoryBound(to: UInt8.self), count: extent.width * 8) }
                let normalized = try Validation.rgba(rgba, expected: expected.checkpoints.last!, precision: precision)
                let referenceRGBA = try Validation.normalizedRGBA(expected.rgba, width: extent.width * 2, height: extent.height * 2)
                let quantized = try Validation.rgba(rgba, expected: referenceRGBA, precision: precision)
                let encoder = JSONEncoder()
                let stageJSON = try JSONSerialization.jsonObject(with: encoder.encode(comparisons))
                let normalizedJSON = try JSONSerialization.jsonObject(with: encoder.encode(normalized))
                let quantizedJSON = try JSONSerialization.jsonObject(with: encoder.encode(quantized))
                let prefix = precision.rawValue
                var artifacts: [[String: Any]] = []
                artifacts.append(try artifact(prefix + "-input.f32le", Array(UnsafeBufferPointer(start: pointer, count: extent.pixels * 4))))
                artifacts.append(try artifact(prefix + "-reference-input.f32le", planar))
                for (observed, trusted) in zip(actual, expected.checkpoints) {
                    artifacts.append(try artifact(prefix + "-" + observed.name + ".f32le", observed.values))
                    artifacts.append(try artifact(prefix + "-reference-" + observed.name + ".f32le", trusted.values))
                }
                for (name, data) in [(prefix + "-output.rgba8", rgba), (prefix + "-reference-output.rgba8", expected.rgba)] {
                    try data.write(to: directory.appendingPathComponent(name), options: .withoutOverwriting)
                    artifacts.append(["file": name, "bytes": data.count, "sha256": sha256(data)])
                }
                runs.append(["precision": precision.rawValue, "inputMaximumError": inputMax, "inputMeanError": inputSum / Double(extent.pixels * 3),
                    "invalidOutputFlags": integrated.validationFlagsAfterCompletion(),
                    "inputFailures": inputFailures, "stages": stageJSON, "rgbaVsFloat": normalizedJSON, "rgbaVsRGBA": quantizedJSON,
                    "integratedRGBAHash": sha256(rgba), "referenceRGBAHash": sha256(expected.rgba), "shaderSHA": integrated.shaderSha256,
                    "artifacts": artifacts, "shape": [extent.width, extent.height], "stageLayout": "CHW", "inputLayout": "HWC4", "referenceInputLayout": "CHW",
                    "outcome": inputFailures == 0 && comparisons.allSatisfy(\.passed) && normalized.passed && quantized.passed ? "PASS" : "FAIL"])
            }
        }
        let metadata = try JSONSerialization.jsonObject(with: JSONEncoder().encode(color))
        let result: [String: Any] = ["schema": "aethervsr.m13.phase2-paused-parity/1", "pts": frame.pts, "color": metadata,
            "modelSHA": ProductionModel.fileHash, "planes": planes, "runs": runs, "outcome": runs.allSatisfy { $0["outcome"] as? String == "PASS" } ? "PASS" : "FAIL"]
        try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]).write(to: directory.appendingPathComponent("result.json"), options: .withoutOverwriting)
        return result
    }

    static func reference(_ buffer: CVPixelBuffer, color: VideoColor) -> [Double] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly); defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let width = color.width, height = color.height
        var output = [Double](repeating: 0, count: width * height * 4)
        let yuv = color.format != kCVPixelFormatType_32BGRA
        let luma = (yuv ? CVPixelBufferGetBaseAddressOfPlane(buffer, 0) : CVPixelBufferGetBaseAddress(buffer))!.assumingMemoryBound(to: UInt8.self)
        let lumaStride = yuv ? CVPixelBufferGetBytesPerRowOfPlane(buffer, 0) : CVPixelBufferGetBytesPerRow(buffer)
        let uv = yuv ? CVPixelBufferGetBaseAddressOfPlane(buffer, 1)!.assumingMemoryBound(to: UInt8.self) : luma
        let uvStride = yuv ? CVPixelBufferGetBytesPerRowOfPlane(buffer, 1) : 0
        let video = color.format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, centered = color.chroma == kCVImageBufferChromaLocation_Center as String
        for row in 0..<height { for column in 0..<width {
            var red: Double, green: Double, blue: Double
            if !yuv {
                let index = row * lumaStride + column * 4
                red = Double(luma[index + 2]) / 255; green = Double(luma[index + 1]) / 255; blue = Double(luma[index]) / 255
            } else {
                let horizontal = Double(column) / 2 - (centered ? 0.25 : 0), vertical = Double(row) / 2 - 0.25
                let left = Int(floor(horizontal)), top = Int(floor(vertical)), alpha = horizontal - floor(horizontal), beta = vertical - floor(vertical)
                func chroma(_ channel: Int) -> Double {
                    func code(_ column: Int, _ row: Int) -> Double { Double(uv[min(height / 2 - 1, max(0, row)) * uvStride + min(width / 2 - 1, max(0, column)) * 2 + channel]) }
                    return (code(left, top) * (1 - alpha) + code(left + 1, top) * alpha) * (1 - beta) + (code(left, top + 1) * (1 - alpha) + code(left + 1, top + 1) * alpha) * beta
                }
                let luminance = (Double(luma[row * lumaStride + column]) - (video ? 16 : 0)) / (video ? 219 : 255)
                let cb = (chroma(0) - 128) / (video ? 224 : 255), cr = (chroma(1) - 128) / (video ? 224 : 255)
                red = luminance + 2 * (1 - 0.2126) * cr; blue = luminance + 2 * (1 - 0.0722) * cb
                green = (luminance - 0.2126 * red - 0.0722 * blue) / 0.7152
            }
            output[(row * width + column) * 4] = min(1, max(0, red)); output[(row * width + column) * 4 + 1] = min(1, max(0, green))
            output[(row * width + column) * 4 + 2] = min(1, max(0, blue))
        } }
        return output
    }
}