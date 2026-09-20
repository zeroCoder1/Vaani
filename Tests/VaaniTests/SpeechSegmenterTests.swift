//
//  SpeechSegmenterTests.swift
//  VaaniTests
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import XCTest
@testable import Vaani

final class SpeechSegmenterTests: XCTestCase {

    /// Declares speech whenever the frame is loud, with no hysteresis, so the
    /// tests exercise the segmenter rather than the detector.
    private final class ThresholdDetector: VoiceActivityDetector, @unchecked Sendable {
        func isSpeech(_ frame: ArraySlice<Float>) -> Bool {
            frame.contains { abs($0) > 0.05 }
        }
        func reset() {}
    }

    private func silence(_ seconds: Double) -> [Float] {
        [Float](repeating: 0, count: Int(16_000 * seconds))
    }

    private func speech(_ seconds: Double) -> [Float] {
        (0..<Int(16_000 * seconds)).map { 0.4 * sin(2 * .pi * 300 * Float($0) / 16_000) }
    }

    private func makeSegmenter() -> SpeechSegmenter {
        SpeechSegmenter(detector: ThresholdDetector())
    }

    func testSilenceProducesNothing() {
        XCTAssertTrue(makeSegmenter().append(silence(3)).isEmpty)
    }

    func testOneUtteranceProducesOneSegment() {
        let segmenter = makeSegmenter()
        var events = segmenter.append(silence(0.5))
        events += segmenter.append(speech(1.5))
        events += segmenter.append(silence(1.0))

        XCTAssertEqual(events.filter { $0 == .speechStarted }.count, 1)
        XCTAssertEqual(events.filter { $0 == .speechEnded }.count, 1)
        let segments = events.compactMap { if case .segment(let s) = $0 { return s } else { return nil } }
        XCTAssertEqual(segments.count, 1)

        // 1.5s of speech, plus pre-roll and the trailing silence that closes it
        let duration = Double(segments[0].count) / 16_000
        XCTAssertGreaterThan(duration, 1.5)
        XCTAssertLessThan(duration, 2.6)
    }

    func testTwoUtterancesSeparatedByAPauseProduceTwoSegments() {
        let segmenter = makeSegmenter()
        var events = segmenter.append(silence(0.4))
        events += segmenter.append(speech(1.0))
        events += segmenter.append(silence(1.0))
        events += segmenter.append(speech(1.0))
        events += segmenter.append(silence(1.0))

        let segments = events.compactMap { if case .segment(let s) = $0 { return s } else { return nil } }
        XCTAssertEqual(segments.count, 2)
    }

    func testShortBlipIsDiscarded() {
        let segmenter = makeSegmenter()
        var events = segmenter.append(silence(0.4))
        events += segmenter.append(speech(0.1))     // under minimumSpeechDuration
        events += segmenter.append(silence(1.0))

        let segments = events.compactMap { if case .segment(let s) = $0 { return s } else { return nil } }
        XCTAssertTrue(segments.isEmpty, "a 0.1s blip should not become an utterance")
        XCTAssertTrue(events.contains(.speechEnded), "but the state should still close")
    }

    func testUnbrokenSpeechIsCutAtTheMaximum() {
        var config = SpeechSegmenter.Configuration()
        config.maximumSegmentDuration = 2
        let segmenter = SpeechSegmenter(detector: ThresholdDetector(), configuration: config)

        let events = segmenter.append(speech(5))
        let segments = events.compactMap { if case .segment(let s) = $0 { return s } else { return nil } }
        XCTAssertGreaterThanOrEqual(segments.count, 2, "5s of unbroken speech should be cut")
        for s in segments {
            XCTAssertLessThanOrEqual(Double(s.count) / 16_000, 2.1)
        }
    }

    func testLeadingAudioIsPreserved() {
        let segmenter = makeSegmenter()
        _ = segmenter.append(silence(1.0))
        let events = segmenter.append(speech(1.0)) + segmenter.append(silence(1.0))
        let segments = events.compactMap { if case .segment(let s) = $0 { return s } else { return nil } }
        XCTAssertEqual(segments.count, 1)
        // pre-roll means the segment starts before the first loud sample
        XCTAssertGreaterThan(segments[0].count, Int(16_000 * 1.0))
    }

    func testFlushReturnsSpeechInProgress() {
        let segmenter = makeSegmenter()
        _ = segmenter.append(silence(0.3))
        _ = segmenter.append(speech(1.0))
        let tail = segmenter.flush()
        XCTAssertNotNil(tail, "stopping mid-utterance should not lose it")
        XCTAssertGreaterThan(tail!.count, Int(16_000 * 0.9))
    }

    func testFlushReturnsNothingWhenIdle() {
        let segmenter = makeSegmenter()
        _ = segmenter.append(silence(1.0))
        XCTAssertNil(segmenter.flush())
    }
}

extension SpeechSegmenterTests {

    func testMidSentencePauseIsAPhraseBoundaryNotASegment() {
        let segmenter = makeSegmenter()
        var events = segmenter.append(silence(0.4))
        events += segmenter.append(speech(1.0))
        events += segmenter.append(silence(0.3))   // hesitation, under silenceDuration
        events += segmenter.append(speech(1.0))
        events += segmenter.append(silence(1.0))   // real end

        // Two: the hesitation, and the closing silence as it passes the phrase
        // threshold on its way to the longer one. The second is deliberate —
        // at 0.22s there is no way to tell a hesitation from a full stop, and
        // transcribing there shows text sooner. The final supersedes it.
        XCTAssertEqual(events.filter { $0 == .phraseBoundary }.count, 2)
        let segments = events.compactMap { if case .segment(let s) = $0 { return s } else { return nil } }
        XCTAssertEqual(segments.count, 1, "both halves belong to one utterance")
        XCTAssertGreaterThan(Double(segments[0].count) / 16_000, 2.0)
    }

    func testPhraseBoundaryFiresOncePerPauseNotPerSilentFrame() {
        let segmenter = makeSegmenter()
        var events = segmenter.append(silence(0.4))
        events += segmenter.append(speech(0.8))
        events += segmenter.append(silence(0.4))   // one long-ish hesitation
        events += segmenter.append(speech(0.8))
        events += segmenter.append(silence(1.0))

        // One per pause, not one per silent frame: the 0.4s hesitation is 20
        // frames past the threshold and must still emit exactly once, plus one
        // for the closing silence.
        XCTAssertEqual(events.filter { $0 == .phraseBoundary }.count, 2)
    }

    func testNoPhraseBoundaryBeforeEnoughSpeech() {
        let segmenter = makeSegmenter()
        var events = segmenter.append(silence(0.4))
        events += segmenter.append(speech(0.1))    // below minimumSpeechDuration
        events += segmenter.append(silence(0.3))
        XCTAssertFalse(events.contains(.phraseBoundary),
                       "a blip followed by a pause is not a phrase")
    }

    func testSeveralPhrasesInOneUtterance() {
        let segmenter = makeSegmenter()
        var events = segmenter.append(silence(0.4))
        for _ in 0..<3 {
            events += segmenter.append(speech(0.8))
            events += segmenter.append(silence(0.3))
        }
        events += segmenter.append(silence(1.0))

        // Three, not four: the last hesitation runs straight into the closing
        // silence, so they are one continuous pause that emits a single
        // boundary before closing the utterance.
        XCTAssertEqual(events.filter { $0 == .phraseBoundary }.count, 3)
        let segments = events.compactMap { if case .segment(let s) = $0 { return s } else { return nil } }
        XCTAssertEqual(segments.count, 1, "still one utterance")
    }
}
