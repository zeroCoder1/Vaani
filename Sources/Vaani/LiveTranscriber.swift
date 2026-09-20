//
//  LiveTranscriber.swift
//  Vaani
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

#if os(iOS)
import Foundation

/// Transcribes speech as it is spoken, by cutting the microphone stream into
/// utterances and transcribing each one.
///
///     let live = LiveTranscriber(recognizer: asr)
///     for await update in try await live.start() {
///         switch update {
///         case .partial(let text): draft = text
///         case .final(let result): transcript += result.text + " "
///         default: break
///         }
///     }
///
/// This is not streaming recognition. IndicConformer is a full-context model
/// with no cached state, so it transcribes whole utterances; there is no way to
/// feed it 200 ms and get incremental output. What this does instead is wait
/// for a pause, then transcribe what was said. Latency is therefore the length
/// of the pause plus processing, which at a real-time factor near 0.05 is a
/// small fraction of the utterance.
///
/// Enable ``Configuration/partialInterval`` to also re-transcribe the utterance
/// in progress, which gives text that updates while someone is still speaking
/// at the cost of extra work. Those results are provisional and will change;
/// present them differently from finals.
public final class LiveTranscriber: @unchecked Sendable {

    /// Provisional text for the utterance in progress.
    public struct Partial: Sendable {
        /// Everything recognised so far. Later words may still change.
        public let text: String
        /// The leading words unchanged since the previous provisional result.
        /// Successive passes revise the tail far more than the start, so this
        /// prefix can be shown as settled while the rest is styled as a draft.
        public let stablePrefix: String
    }

    public enum Update: Sendable {
        /// Someone started speaking.
        case speechStarted
        /// Provisional text for the utterance in progress. Will change.
        case partial(Partial)
        /// A finished utterance. Settled.
        case final(SpeechRecognizer.Transcription)
        /// The utterance ended; no further partials until the next one.
        case speechEnded
        /// Transcription failed for one segment. The stream continues.
        case failed(String)
    }

    public struct Configuration: Sendable {
        /// Silence that closes an utterance.
        public var silenceDuration: TimeInterval = 0.6
        /// Utterances shorter than this are treated as noise and dropped.
        public var minimumSpeechDuration: TimeInterval = 0.3
        /// Hard cut, so someone who never pauses still gets transcribed.
        public var maximumSegmentDuration: TimeInterval = 15
        /// Longest gap between provisional results while someone keeps
        /// talking without pausing. Zero disables them, roughly halving the
        /// work. Provisional results are also produced at every phrase pause,
        /// which is usually sooner than this.
        public var partialInterval: TimeInterval = 1.5
        /// Multiple of the previous provisional result's cost to wait before
        /// starting another. Re-transcribing re-runs the whole utterance, so
        /// the cost grows as someone speaks; without this, long sentences would
        /// spend every spare cycle on drafts and starve the final.
        public var partialBackoff: Double = 2.0
        /// Decoder for both provisional and final text.
        public var decoder: SpeechRecognizer.Decoder = .ctc

        public init() {}
    }

    /// Injected so tests can drive the orchestration without a model.
    typealias TranscribeStep = @Sendable ([Float], SpeechRecognizer.Decoder) throws
        -> SpeechRecognizer.Transcription

    private let transcriber: TranscribeStep
    private let configuration: Configuration
    private let microphone: MicrophoneCapture
    private let segmenter: SpeechSegmenter
    private let queue = DispatchQueue(label: "dev.vaani.live")

    private var continuation: AsyncStream<Update>.Continuation?
    private var working = false
    private var lastPartialStarted = Date.distantPast
    private var lastPartialCost: TimeInterval = 0
    private var previousPartialText = ""
    /// Finished utterances waiting for the worker. Never dropped: a partial in
    /// flight must not cost someone a sentence.
    private var queued: [[Float]] = []

    public init(recognizer: SpeechRecognizer,
                detector: VoiceActivityDetector = EnergyVoiceActivityDetector(),
                configuration: Configuration = Configuration(),
                microphone: MicrophoneCapture = MicrophoneCapture()) {
        self.transcriber = { samples, decoder in
            try recognizer.transcribe(samples, using: decoder)
        }
        self.configuration = configuration
        self.microphone = microphone

        var segmenterConfiguration = SpeechSegmenter.Configuration()
        segmenterConfiguration.silenceDuration = configuration.silenceDuration
        segmenterConfiguration.minimumSpeechDuration = configuration.minimumSpeechDuration
        segmenterConfiguration.maximumSegmentDuration = configuration.maximumSegmentDuration
        self.segmenter = SpeechSegmenter(detector: detector,
                                         configuration: segmenterConfiguration)
    }

    /// Test seam: same orchestration, a stubbed transcription step.
    init(transcriber: @escaping TranscribeStep,
         detector: VoiceActivityDetector,
         configuration: Configuration) {
        self.transcriber = transcriber
        self.configuration = configuration
        self.microphone = MicrophoneCapture()

        var segmenterConfiguration = SpeechSegmenter.Configuration()
        segmenterConfiguration.silenceDuration = configuration.silenceDuration
        segmenterConfiguration.minimumSpeechDuration = configuration.minimumSpeechDuration
        segmenterConfiguration.maximumSegmentDuration = configuration.maximumSegmentDuration
        self.segmenter = SpeechSegmenter(detector: detector,
                                         configuration: segmenterConfiguration)
    }

    /// Test seam: a stream that is fed by `feed` rather than the microphone.
    func makeStream() -> AsyncStream<Update> {
        segmenter.reset()
        return AsyncStream<Update>(bufferingPolicy: .unbounded) { continuation in
            self.queue.async { self.continuation = continuation }
        }
    }

    /// Test seam: deliver audio as the microphone would.
    func feed(_ samples: [Float]) {
        queue.sync { ingest(samples) }
    }

    /// Test seam: wait for queued work to drain.
    func drain() {
        while queue.sync(execute: { working || !queued.isEmpty }) {
            Thread.sleep(forTimeInterval: 0.01)
        }
        queue.sync {}
    }

    public var isRunning: Bool { microphone.isRecording }

    /// Request microphone access. Needs `NSMicrophoneUsageDescription`.
    public func requestPermission() async -> Bool {
        await microphone.requestPermission()
    }

    /// Begin listening. The stream finishes when ``stop()`` is called.
    public func start() async throws -> AsyncStream<Update> {
        segmenter.reset()

        let stream = AsyncStream<Update>(bufferingPolicy: .unbounded) { continuation in
            self.queue.async { self.continuation = continuation }
        }

        microphone.stream(retainingSamples: false) { [weak self] chunk in
            guard let self else { return }
            // Hop off the audio thread before doing anything that allocates.
            self.queue.async { self.ingest(chunk) }
        }

        try await microphone.start()
        return stream
    }

    /// Stop listening, transcribing any utterance still in progress.
    public func stop() async {
        _ = await microphone.stop()
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            queue.async {
                if let tail = self.segmenter.flush() {
                    self.transcribe(tail, final: true)
                }
                self.finishWhenDrained()
                done.resume()
            }
        }
    }

    /// Keep the stream open until queued utterances have been transcribed, so
    /// stopping mid-sentence still delivers the last one.
    private func finishWhenDrained() {
        guard !working, queued.isEmpty else {
            queue.asyncAfter(deadline: .now() + 0.1) { [weak self] in
                self?.finishWhenDrained()
            }
            return
        }
        continuation?.finish()
        continuation = nil
    }

    // Everything below runs on `queue`.

    private func ingest(_ chunk: [Float]) {
        for event in segmenter.append(chunk) {
            switch event {
            case .speechStarted:
                continuation?.yield(.speechStarted)
            case .speechEnded:
                continuation?.yield(.speechEnded)
            case .phraseBoundary:
                // The best moment to draft: what came before is a complete
                // phrase, so this transcription is unlikely to be revised.
                emitPartial(force: true)
            case .segment(let samples):
                previousPartialText = ""
                transcribe(samples, final: true)
            }
        }
        emitPartial(force: false)
    }

    private func emitPartial(force: Bool) {
        guard configuration.partialInterval > 0, !working else { return }

        let elapsed = Date().timeIntervalSince(lastPartialStarted)
        // Never start sooner than the previous draft cost, scaled. A 6-second
        // utterance takes longer to re-transcribe than a 1-second one, so the
        // gap widens on its own as someone keeps talking.
        let floor = lastPartialCost * configuration.partialBackoff
        guard elapsed >= (force ? floor : max(configuration.partialInterval, floor)) else {
            return
        }

        let inProgress = segmenter.currentSegment
        guard Double(inProgress.count) / 16_000 >= configuration.minimumSpeechDuration else {
            return
        }
        lastPartialStarted = Date()
        transcribe(inProgress, final: false)
    }

    /// Words shared with the previous draft, which are unlikely to move again.
    private func stablePrefix(of text: String) -> String {
        let new = text.split(separator: " ")
        let old = previousPartialText.split(separator: " ")
        var shared: [Substring] = []
        for (a, b) in zip(old, new) where a == b { shared.append(b) }
        return shared.joined(separator: " ")
    }

    /// Finished utterances are queued; provisional ones are dropped if the
    /// worker is busy, since a newer one will be along shortly.
    private func transcribe(_ samples: [Float], final: Bool) {
        if final {
            queued.append(samples)
            pump()
        } else if !working {
            run(samples, final: false)
        }
    }

    private func pump() {
        guard !working, !queued.isEmpty else { return }
        run(queued.removeFirst(), final: true)
    }

    /// One transcription at a time: the encoder is the expensive part, and two
    /// at once contend for the same cores without finishing any sooner.
    private func run(_ samples: [Float], final: Bool) {
        working = true
        let transcribe = self.transcriber
        let decoder = configuration.decoder

        Task.detached(priority: final ? .userInitiated : .utility) { [weak self] in
            let started = Date()
            var outcome: Result<SpeechRecognizer.Transcription, Error>
            do {
                outcome = .success(try transcribe(samples, decoder))
            } catch {
                outcome = .failure(error)
            }
            let cost = Date().timeIntervalSince(started)

            self?.queue.async {
                guard let self else { return }
                if !final { self.lastPartialCost = cost }

                switch outcome {
                case .success(let result) where !result.text.isEmpty:
                    if final {
                        self.continuation?.yield(.final(result))
                    } else {
                        let partial = Partial(text: result.text,
                                              stablePrefix: self.stablePrefix(of: result.text))
                        self.previousPartialText = result.text
                        self.continuation?.yield(.partial(partial))
                    }
                case .failure(let error) where final:
                    self.continuation?.yield(.failed(error.localizedDescription))
                default:
                    break
                }

                self.working = false
                self.pump()
            }
        }
    }
}
#endif
