//
//  LiveTranscriberTests.swift
//  VaaniTests
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

#if os(iOS)
import XCTest
@testable import Vaani

final class LiveTranscriberTests: XCTestCase {

    private final class ThresholdDetector: VoiceActivityDetector, @unchecked Sendable {
        func isSpeech(_ frame: ArraySlice<Float>) -> Bool { frame.contains { abs($0) > 0.05 } }
        func reset() {}
    }

    private func silence(_ seconds: Double) -> [Float] {
        [Float](repeating: 0, count: Int(16_000 * seconds))
    }

    private func speech(_ seconds: Double) -> [Float] {
        (0..<Int(16_000 * seconds)).map { 0.4 * sin(2 * .pi * 300 * Float($0) / 16_000) }
    }

    /// Counts how many times the stubbed transcriber ran, and returns text
    /// derived from the input length so results are distinguishable.
    private func makeTranscriber(partials: Bool)
        -> (LiveTranscriber, @Sendable () -> Int) {
        let calls = NSMutableArray()
        let lock = NSLock()

        var configuration = LiveTranscriber.Configuration()
        configuration.partialInterval = partials ? 0.2 : 0

        let live = LiveTranscriber(
            transcriber: { samples, _ in
                lock.withLock { calls.add(samples.count) }
                return SpeechRecognizer.Transcription(
                    text: "utterance of \(samples.count) samples",
                    language: .hindi,
                    decoder: .ctc,
                    audioDuration: Double(samples.count) / 16_000,
                    processingTime: 0.01)
            },
            detector: ThresholdDetector(),
            configuration: configuration)

        return (live, { lock.withLock { calls.count } })
    }

    private func collect(_ live: LiveTranscriber,
                         feeding blocks: [[Float]]) async -> [LiveTranscriber.Update] {
        let stream = live.makeStream()
        let collector = Task { () -> [LiveTranscriber.Update] in
            var seen: [LiveTranscriber.Update] = []
            for await update in stream { seen.append(update) }
            return seen
        }
        for block in blocks { live.feed(block) }
        live.drain()
        await live.stop()
        return await collector.value
    }

    private func finals(_ updates: [LiveTranscriber.Update]) -> [String] {
        updates.compactMap { if case .final(let r) = $0 { return r.text } else { return nil } }
    }

    /// The reported failure: with provisional results off, utterances went
    /// missing. Three spoken phrases must produce three finals either way.
    func testEveryUtteranceIsTranscribedWithoutPartials() async {
        let (live, _) = makeTranscriber(partials: false)
        let updates = await collect(live, feeding: [
            silence(0.4), speech(1.0), silence(1.0),
            speech(1.0), silence(1.0),
            speech(1.0), silence(1.0),
        ])
        XCTAssertEqual(finals(updates).count, 3, "no utterance may be dropped")
    }

    func testEveryUtteranceIsTranscribedWithPartials() async {
        let (live, _) = makeTranscriber(partials: true)
        let updates = await collect(live, feeding: [
            silence(0.4), speech(1.0), silence(1.0),
            speech(1.0), silence(1.0),
            speech(1.0), silence(1.0),
        ])
        XCTAssertEqual(finals(updates).count, 3, "no utterance may be dropped")
    }

    func testUtteranceIsNotLostWhileADraftIsRunning() async {
        let (live, _) = makeTranscriber(partials: true)
        let updates = await collect(live, feeding: [
            silence(0.4),
            speech(2.0),      // long enough to trigger drafts
            silence(1.0),     // then finish while one may be in flight
        ])
        XCTAssertEqual(finals(updates).count, 1)
    }

    func testStoppingMidUtteranceStillDeliversIt() async {
        let (live, _) = makeTranscriber(partials: false)
        let updates = await collect(live, feeding: [silence(0.4), speech(1.5)])
        XCTAssertEqual(finals(updates).count, 1, "flush should transcribe the tail")
    }

    func testPartialsDoNotDuplicateFinals() async {
        let (live, _) = makeTranscriber(partials: true)
        let updates = await collect(live, feeding: [
            silence(0.4), speech(1.5), silence(1.0),
        ])
        XCTAssertEqual(finals(updates).count, 1)
        XCTAssertTrue(updates.contains { if case .partial = $0 { return true } else { return false } },
                      "a 1.5s utterance should produce at least one draft")
    }
}
#endif

#if os(iOS)
extension LiveTranscriberTests {

    /// The stub above returns instantly; the real encoder does not. These feed
    /// utterances while a transcription is still running, which is what happens
    /// on device when someone keeps talking.
    private func makeSlowTranscriber(partials: Bool, cost: TimeInterval)
        -> LiveTranscriber {
        var configuration = LiveTranscriber.Configuration()
        configuration.partialInterval = partials ? 0.2 : 0
        // Utterances of equal length would otherwise yield identical text and
        // look like duplicates when deduplicated.
        let counter = NSMutableArray()
        let lock = NSLock()
        return LiveTranscriber(
            transcriber: { samples, _ in
                Thread.sleep(forTimeInterval: cost)
                let n = lock.withLock { counter.add(1); return counter.count }
                return SpeechRecognizer.Transcription(
                    text: "u\(n)-\(samples.count)",
                    language: .hindi, decoder: .ctc,
                    audioDuration: Double(samples.count) / 16_000,
                    processingTime: cost)
            },
            detector: ThresholdDetector(),
            configuration: configuration)
    }

    func testSlowTranscriptionDoesNotDropUtterancesWithoutPartials() async {
        let live = makeSlowTranscriber(partials: false, cost: 0.25)
        let updates = await collect(live, feeding: [
            silence(0.4),
            speech(0.8), silence(0.8),
            speech(0.8), silence(0.8),
            speech(0.8), silence(0.8),
            speech(0.8), silence(0.8),
        ])
        XCTAssertEqual(finals(updates).count, 4,
                       "queued utterances must all be transcribed, however slow")
    }

    func testSlowTranscriptionDoesNotDropUtterancesWithPartials() async {
        let live = makeSlowTranscriber(partials: true, cost: 0.25)
        let updates = await collect(live, feeding: [
            silence(0.4),
            speech(0.8), silence(0.8),
            speech(0.8), silence(0.8),
            speech(0.8), silence(0.8),
            speech(0.8), silence(0.8),
        ])
        XCTAssertEqual(finals(updates).count, 4)
    }

    func testFinalsAreNotDuplicatedWhenQueued() async {
        let live = makeSlowTranscriber(partials: false, cost: 0.3)
        let updates = await collect(live, feeding: [
            silence(0.4),
            speech(0.8), silence(0.8),
            speech(0.8), silence(0.8),
        ])
        let texts = finals(updates)
        XCTAssertEqual(texts.count, 2)
        XCTAssertEqual(Set(texts).count, texts.count, "each utterance exactly once")
    }
}
#endif
