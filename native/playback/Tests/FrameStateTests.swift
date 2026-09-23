@testable import PlaybackCore
import XCTest

final class FrameStateTests: XCTestCase {
    @MainActor func testSeekCompletionIsOneShot() {
        var results: [Bool] = []
        let completion = SeekCompletion { results.append($0) }
        completion.finish(true); completion.finish(false); completion.finish(true)
        XCTAssertEqual(results, [true])
    }
    func testGenerationsCapacityAndDuplicates() throws {
        var state = FrameState()
        XCTAssertNil(state.acquire(pts: 0))
        state.invalidate(playing: true)
        let first = try XCTUnwrap(state.acquire(pts: 1)), second = try XCTUnwrap(state.acquire(pts: 2))
        XCTAssertNil(state.acquire(pts: 2)); XCTAssertNil(state.acquire(pts: 3)); XCTAssertEqual(state.active.count, 2)
        state.invalidate(playing: false); XCTAssertFalse(state.accepts(first)); XCTAssertNil(state.acquire(pts: 3))
        state.release(first); state.release(second); XCTAssertTrue(state.active.isEmpty)
        state.invalidate(playing: true)
        let afterSeek = try XCTUnwrap(state.acquire(pts: 0.5)); XCTAssertTrue(state.accepts(afterSeek)); XCTAssertFalse(state.accepts(second))
        XCTAssertNil(state.acquire(pts: .nan)); XCTAssertNil(state.acquire(pts: .infinity)); XCTAssertNil(state.acquire(pts: -1))
        state.fail(); XCTAssertFalse(state.accepts(afterSeek)); state.invalidate(playing: true); XCTAssertNil(state.acquire(pts: 1))
        state.release(afterSeek); XCTAssertTrue(state.active.isEmpty)
    }
}