//
//  SpeechSegmenter.swift
//  Vaani
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import Foundation

/// Cuts a running audio stream into utterances on pauses.
///
/// The model transcribes whole utterances rather than a stream, so something
/// has to decide where one ends. This walks the stream in fixed frames, asks a
/// ``VoiceActivityDetector`` about each, and emits a segment once speech has
/// been followed by enough silence.
final class SpeechSegmenter {

    struct Configuration {
        var sampleRate: Double = 16_000
        var frameDuration: TimeInterval = 0.02
        /// Silence after speech that closes a segment. Roughly a sentence or
        /// turn boundary.
        var silenceDuration: TimeInterval = 0.6
        /// Shorter silence that marks a phrase boundary without ending the
        /// utterance. People pause mid-sentence to think, and those pauses are
        /// the best moment to show text: what came before is complete, so a
        /// transcription taken here is unlikely to be revised.
        var phrasePauseDuration: TimeInterval = 0.22
        /// Segments shorter than this are discarded as noise.
        var minimumSpeechDuration: TimeInterval = 0.3
        /// Hard cut, so someone who never pauses still gets transcribed.
        var maximumSegmentDuration: TimeInterval = 15
        /// Audio kept before speech starts, so the first phoneme is not clipped.
        var leadingPadding: TimeInterval = 0.2
    }

    enum Event: Equatable {
        case speechStarted
        /// A mid-sentence pause. The utterance continues.
        case phraseBoundary
        case segment([Float])
        case speechEnded
    }

    private let configuration: Configuration
    private let detector: VoiceActivityDetector

    private var pending: [Float] = []      // samples not yet formed into a frame
    private var segment: [Float] = []      // current utterance
    private var preRoll: [Float] = []      // rolling pre-speech context
    private var speaking = false
    private var silentFrames = 0
    /// Counted separately from the segment: the segment also holds pre-roll and
    /// trailing silence, which together exceed the minimum on their own and
    /// would let every cough through.
    private var speechFrames = 0
    private var announcedPhrase = false

    private var frameLength: Int { Int(configuration.sampleRate * configuration.frameDuration) }
    private var silenceFrameCount: Int {
        max(1, Int(configuration.silenceDuration / configuration.frameDuration))
    }
    private var phraseFrameCount: Int {
        max(1, Int(configuration.phrasePauseDuration / configuration.frameDuration))
    }
    private var preRollLimit: Int { Int(configuration.sampleRate * configuration.leadingPadding) }
    private var maximumSamples: Int {
        Int(configuration.sampleRate * configuration.maximumSegmentDuration)
    }
    private var minimumSpeechFrames: Int {
        max(1, Int(configuration.minimumSpeechDuration / configuration.frameDuration))
    }

    /// Duration of the utterance in progress, for interim transcription.
    var currentSegmentDuration: TimeInterval {
        Double(segment.count) / configuration.sampleRate
    }

    var currentSegment: [Float] { segment }

    init(detector: VoiceActivityDetector, configuration: Configuration = Configuration()) {
        self.detector = detector
        self.configuration = configuration
    }

    func reset() {
        pending.removeAll(keepingCapacity: true)
        segment.removeAll(keepingCapacity: true)
        preRoll.removeAll(keepingCapacity: true)
        speaking = false
        silentFrames = 0
        speechFrames = 0
        announcedPhrase = false
        detector.reset()
    }

    func append(_ samples: [Float]) -> [Event] {
        pending.append(contentsOf: samples)
        var events: [Event] = []

        while pending.count >= frameLength {
            let frame = Array(pending.prefix(frameLength))
            pending.removeFirst(frameLength)
            events.append(contentsOf: consume(frame))
        }
        return events
    }

    private func consume(_ frame: [Float]) -> [Event] {
        var events: [Event] = []
        let isSpeech = detector.isSpeech(frame[...])

        if isSpeech {
            if !speaking {
                speaking = true
                // Start from the pre-roll so the utterance does not begin
                // mid-word: the detector needs a few frames to confirm.
                segment = preRoll
                preRoll.removeAll(keepingCapacity: true)
                events.append(.speechStarted)
            }
            silentFrames = 0
            announcedPhrase = false
            speechFrames += 1
            segment.append(contentsOf: frame)
        } else if speaking {
            silentFrames += 1
            // Keep the trailing silence; it carries the final phoneme's decay.
            segment.append(contentsOf: frame)
            if silentFrames >= silenceFrameCount {
                events.append(contentsOf: close())
            } else if silentFrames == phraseFrameCount, !announcedPhrase,
                      speechFrames >= minimumSpeechFrames {
                // Fires once per pause, including the closing one: at this
                // point a hesitation and a full stop are indistinguishable, and
                // transcribing now shows text sooner. If it does turn out to be
                // the end, the segment that follows supersedes the partial.
                announcedPhrase = true
                events.append(.phraseBoundary)
            }
        } else {
            preRoll.append(contentsOf: frame)
            if preRoll.count > preRollLimit {
                preRoll.removeFirst(preRoll.count - preRollLimit)
            }
        }

        if speaking && segment.count >= maximumSamples {
            events.append(contentsOf: close())
        }
        return events
    }

    private func close() -> [Event] {
        defer {
            segment.removeAll(keepingCapacity: true)
            speaking = false
            silentFrames = 0
            speechFrames = 0
            announcedPhrase = false
        }
        var events: [Event] = [.speechEnded]
        if speechFrames >= minimumSpeechFrames {
            events.insert(.segment(segment), at: 0)
        }
        return events
    }

    /// Emit whatever is in progress, for when recording stops mid-utterance.
    func flush() -> [Float]? {
        defer { reset() }
        guard speaking, speechFrames >= minimumSpeechFrames else { return nil }
        return segment
    }
}
