//
//  VoiceActivityDetector.swift
//  Vaani
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import Accelerate
import Foundation

/// Decides which parts of an audio stream contain speech.
///
/// ``LiveTranscriber`` uses this to find pauses, since the model transcribes
/// whole utterances rather than a running stream. Supply your own conforming
/// type to swap in a learned detector.
public protocol VoiceActivityDetector: AnyObject, Sendable {
    /// - Parameter frame: contiguous mono samples at 16 kHz.
    /// - Returns: whether this frame is speech.
    func isSpeech(_ frame: ArraySlice<Float>) -> Bool

    /// Forget accumulated state, e.g. between recordings.
    func reset()
}

/// Energy-based detector with an adaptive noise floor.
///
/// The floor is the quietest frame in a recent window rather than a running
/// average, which matters: an average cannot distinguish sustained speech from
/// a room that got louder, so it either chases the speaker until it goes deaf
/// or latches onto noise and hears it as speech.
///
/// It costs a handful of arithmetic per frame and needs no extra model, and is
/// good enough in ordinary rooms. Two limitations are worth knowing:
///
/// - It only measures level, not whether a sound resembles speech, so sustained
///   background noise at speech-like volume will read as speech.
/// - The estimate depends on the window containing something quiet. Real speech
///   supplies that in the gaps between words. A signal with genuinely constant
///   amplitude — a tone, an engine — is ambiguous by construction, and no
///   level-only detector can resolve it.
///
/// For noisy environments, conform your own type to ``VoiceActivityDetector``
/// around a learned detector such as Silero VAD and pass it to
/// ``LiveTranscriber``; the protocol exists for exactly that.
public final class EnergyVoiceActivityDetector: VoiceActivityDetector, @unchecked Sendable {

    public struct Configuration: Sendable {
        /// How far above the noise floor a frame must sit to count as speech.
        public var activationRatio: Float = 3.0
        /// Absolute floor, so near-silence never registers however quiet the
        /// room, and so startup is not deaf before the floor is established.
        public var absoluteFloor: Float = 1e-4
        /// Consecutive frames required before the decision flips. Suppresses
        /// chatter on the boundary in both directions.
        public var framesToConfirm: Int = 3
        /// How much recent audio the noise floor is estimated from. It needs to
        /// span the gaps between words, since those gaps are what the estimate
        /// relies on.
        public var noiseWindow: TimeInterval = 5.0
        /// Frame length the caller feeds, used to size the window.
        public var frameDuration: TimeInterval = 0.02

        public init() {}
    }

    private let configuration: Configuration
    private let lock = NSLock()
    private var history: [Float] = []
    private var cursor = 0
    private var speaking = false
    private var pending = 0

    public init(configuration: Configuration = Configuration()) {
        self.configuration = configuration
    }

    private var windowSize: Int {
        max(8, Int(configuration.noiseWindow / configuration.frameDuration))
    }

    public func reset() {
        lock.withLock {
            history.removeAll(keepingCapacity: true)
            cursor = 0
            speaking = false
            pending = 0
        }
    }

    public func isSpeech(_ frame: ArraySlice<Float>) -> Bool {
        guard !frame.isEmpty else { return false }

        var rms: Float = 0
        frame.withUnsafeBufferPointer {
            vDSP_rmsqv($0.baseAddress!, 1, &rms, vDSP_Length($0.count))
        }

        return lock.withLock {
            // The floor is the quietest recent frame rather than a running
            // average. An average cannot tell sustained speech from a room that
            // got louder, so it either chases the speaker and goes deaf, or
            // latches onto noise and hears it as speech. A minimum cannot: talk
            // as long as you like, the pauses are still the quietest thing in
            // the window.
            if history.count < windowSize {
                history.append(rms)
            } else {
                history[cursor] = rms
                cursor = (cursor + 1) % windowSize
            }

            // Until enough history exists, fall back to the absolute floor so
            // speech starting at the very first frame is still heard.
            let floor = history.count >= 8
                ? max(history.min() ?? 0, configuration.absoluteFloor)
                : configuration.absoluteFloor
            let loud = rms > floor * configuration.activationRatio

            if loud == speaking {
                pending = 0
            } else {
                pending += 1
                if pending >= configuration.framesToConfirm {
                    speaking = loud
                    pending = 0
                }
            }
            return speaking
        }
    }
}
