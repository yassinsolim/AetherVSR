import Foundation

public struct FrameIdentity: Equatable, Codable, Sendable {
    public let generation: UInt64
    public let sequence: UInt64
    public let presentationTime: Double
}

public struct FrameState {
    public private(set) var generation: UInt64 = 0
    public private(set) var sequence: UInt64 = 0
    public private(set) var lastPTS: Double?
    public private(set) var terminal = false
    public private(set) var playing = false
    public private(set) var active: Set<UInt64> = []
    public let capacity = 2

    public init() {}
    public mutating func invalidate(playing: Bool) {
        generation &+= 1; lastPTS = nil; self.playing = playing && !terminal
    }
    public mutating func fail() { terminal = true; invalidate(playing: false) }
    public mutating func setPlaying(_ value: Bool) { playing = value && !terminal }
    public mutating func acquire(pts: Double, pausedDiagnostic: Bool = false) -> FrameIdentity? {
        guard !terminal && (playing || pausedDiagnostic) && pts.isFinite && pts >= 0,
              lastPTS.map({ pts > $0 + 0.000001 }) ?? true, active.count < capacity else { return nil }
        sequence &+= 1; lastPTS = pts; active.insert(sequence)
        return FrameIdentity(generation: generation, sequence: sequence, presentationTime: pts)
    }
    public func accepts(_ identity: FrameIdentity) -> Bool { !terminal && generation == identity.generation }
    public mutating func release(_ identity: FrameIdentity) { active.remove(identity.sequence) }
}