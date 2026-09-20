//
//  VoiceActivityDetectorTests.swift
//  VaaniTests
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import XCTest
@testable import Vaani

final class VoiceActivityDetectorTests: XCTestCase {

    private let frameLength = 320   // 20 ms at 16 kHz

    private func tone(_ amplitude: Float, frames: Int) -> [Float] {
        (0..<(frames * frameLength)).map {
            amplitude * sin(2 * .pi * 300 * Float($0) / 16_000)
        }
    }

    /// Speech-shaped: a tone broken by short gaps, as words are. A constant
    /// tone is not a fair stand-in, since the quiet between words is precisely
    /// what the noise floor is estimated from.
    private func spokenWords(_ amplitude: Float, frames: Int) -> [Float] {
        (0..<(frames * frameLength)).map { i in
            let frame = i / frameLength
            let inGap = frame % 10 >= 8          // ~160ms of speech, ~40ms gap
            let envelope: Float = inGap ? 0.02 : 1
            return amplitude * envelope * sin(2 * .pi * 300 * Float(i) / 16_000)
        }
    }

    private func noise(_ amplitude: Float, frames: Int, seed: UInt64 = 1) -> [Float] {
        var state = seed
        return (0..<(frames * frameLength)).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return (Float(state >> 40) / Float(1 << 24) - 0.5) * 2 * amplitude
        }
    }

    /// Feed a signal frame by frame, returning the decision for each frame.
    private func run(_ vad: VoiceActivityDetector, _ samples: [Float]) -> [Bool] {
        stride(from: 0, to: samples.count - frameLength + 1, by: frameLength).map {
            vad.isSpeech(samples[$0..<($0 + frameLength)])
        }
    }

    func testSilenceIsNotSpeech() {
        let vad = EnergyVoiceActivityDetector()
        XCTAssertFalse(run(vad, [Float](repeating: 0, count: frameLength * 50)).contains(true))
    }

    func testLoudSpeechAfterSilenceIsDetected() {
        let vad = EnergyVoiceActivityDetector()
        _ = run(vad, noise(0.0005, frames: 25))          // quiet room
        let decisions = run(vad, tone(0.3, frames: 25))  // someone speaks
        XCTAssertTrue(decisions.dropFirst(5).allSatisfy { $0 },
                      "sustained speech should register once confirmed")
    }

    func testSingleLoudFrameDoesNotFlipTheDecision() {
        var config = EnergyVoiceActivityDetector.Configuration()
        config.framesToConfirm = 3
        let vad = EnergyVoiceActivityDetector(configuration: config)
        _ = run(vad, noise(0.0005, frames: 25))

        // one loud frame, then quiet again
        var burst = [Float](repeating: 0.0005, count: frameLength * 3)
        burst.replaceSubrange(frameLength..<(frameLength * 2),
                              with: tone(0.5, frames: 1))
        XCTAssertFalse(run(vad, burst).contains(true),
                       "a single frame should not be enough to declare speech")
    }

    func testSpeechIsFoundAboveElevatedBackgroundNoise() {
        let vad = EnergyVoiceActivityDetector()
        _ = run(vad, noise(0.02, frames: 50))            // noisy room
        let decisions = run(vad, tone(0.4, frames: 25))
        XCTAssertTrue(decisions.dropFirst(5).contains(true),
                      "the floor should adapt so louder speech still stands out")
    }

    func testTrailingSilenceEndsSpeech() {
        let vad = EnergyVoiceActivityDetector()
        _ = run(vad, noise(0.0005, frames: 25))
        _ = run(vad, tone(0.3, frames: 25))
        let decisions = run(vad, noise(0.0005, frames: 25))
        XCTAssertFalse(decisions.last!, "speech should end when the level drops")
    }

    func testResetClearsState() {
        let vad = EnergyVoiceActivityDetector()
        _ = run(vad, noise(0.0005, frames: 25))
        _ = run(vad, tone(0.3, frames: 25))
        vad.reset()
        XCTAssertFalse(run(vad, [Float](repeating: 0, count: frameLength * 10)).contains(true))
    }
}

extension VoiceActivityDetectorTests {

    /// The failure that prompted the sliding-minimum floor: tapping Start and
    /// talking immediately calibrated the floor to speech level, leaving the
    /// detector deaf until the speaker stopped.
    func testSpeechStartingAtTheVeryFirstFrameIsHeard() {
        let vad = EnergyVoiceActivityDetector()
        let decisions = run(vad, spokenWords(0.3, frames: 100))
        let heard = decisions.filter { $0 }.count
        XCTAssertGreaterThan(heard, 60,
                             "speech from frame zero should be heard, not calibrated away")
    }

    func testSustainedSpeechDoesNotGraduallyGoDeaf() {
        let vad = EnergyVoiceActivityDetector()
        _ = run(vad, noise(0.0005, frames: 25))
        // 6 seconds of continuous talking, longer than the noise window
        let decisions = run(vad, spokenWords(0.3, frames: 300))
        XCTAssertGreaterThan(decisions.suffix(100).filter { $0 }.count, 60,
                             "a long utterance must not drag the floor up behind it")
    }

    func testSteadyBackgroundNoiseIsNotHeardAsSpeech() {
        let vad = EnergyVoiceActivityDetector()
        let decisions = run(vad, noise(0.02, frames: 200))
        let heard = decisions.suffix(100).filter { $0 }.count
        XCTAssertLessThan(heard, 5, "constant noise should settle into the floor")
    }

    /// Documents the limitation rather than hiding it: a level-only detector
    /// cannot tell a constant tone from constant noise, because the window
    /// never contains anything quieter to compare against.
    func testConstantAmplitudeToneIsAKnownLimitation() {
        let vad = EnergyVoiceActivityDetector()
        let decisions = run(vad, tone(0.3, frames: 100))
        XCTAssertLessThan(decisions.filter { $0 }.count, 30,
                          "if this starts passing, the estimator changed shape")
    }

    func testAlternatingSpeechAndSilenceTracksBoth() {
        let vad = EnergyVoiceActivityDetector()
        _ = run(vad, noise(0.001, frames: 25))
        for _ in 0..<3 {
            XCTAssertTrue(run(vad, spokenWords(0.3, frames: 40)).suffix(20).contains(true))
            XCTAssertFalse(run(vad, noise(0.001, frames: 40)).suffix(20).contains(true))
        }
    }
}
