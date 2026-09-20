//
//  RecognizerModel.swift
//  VaaniDemo
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import Foundation
import Vaani
import SwiftUI

@MainActor
final class RecognizerModel: ObservableObject {

    enum State: Equatable {
        case needsModel
        case downloading(file: String, fraction: Double, index: Int, count: Int)
        case loading
        case ready
        case working
        case failed(String)
    }

    struct Outcome: Identifiable {
        let id = UUID()
        let source: String
        let text: String
        let language: Language
        let decoder: SpeechRecognizer.Decoder
        let realTimeFactor: Double
        let duration: TimeInterval
        let wordErrorRate: Double?
    }

    @Published private(set) var state: State = .needsModel
    @Published private(set) var outcomes: [Outcome] = []
    @Published private(set) var isRecording = false
    @Published private(set) var cachedBytes: Int64 = 0
    @Published private(set) var benchmarkRuns: [BenchmarkRun] = []
    @Published private(set) var benchmarkProgress: String?
    @Published private(set) var isLive = false
    @Published private(set) var liveTranscript = ""
    @Published private(set) var livePartial = ""
    @Published private(set) var liveStable = ""
    @Published private(set) var liveStatus = ""
    @Published var livePartialsEnabled = false

    @Published var language: Language = .hindi { didSet { invalidate(oldValue != language) } }
    @Published var decoder: SpeechRecognizer.Decoder = .ctc {
        didSet { invalidate(oldValue != decoder) }
    }
    /// Default host for the demo. Override it in the app's Settings section, or
    /// point it at `tools/dev.sh serve` when testing changes to the model.
    @AppStorage("modelSource") var source = RecognizerModel.defaultSource
    static let defaultSource = "https://model.supr.works"

    let samples = Sample.bundled

    @Published private var downloadedFromHost: String?
    private var recognizer: SpeechRecognizer?
    private var loaded: (Language, SpeechRecognizer.Decoder)?
    private let microphone = MicrophoneCapture()
    private var live: LiveTranscriber?
    private var liveTask: Task<Void, Never>?

    /// Where the cached files came from. Side-loading with
    /// `tools/dev.sh push-model` leaves no download record, and it is otherwise
    /// a mystery why a model is present without ever having been fetched.
    var origin: String {
        downloadedFromHost.map { "Downloaded from \($0)" } ?? "Side-loaded, not downloaded"
    }

    var isBusy: Bool {
        switch state {
        case .downloading, .loading, .working: true
        default: false
        }
    }

    private func invalidate(_ changed: Bool) {
        guard changed, recognizer != nil else { return }
        recognizer = nil
        loaded = nil
        state = .needsModel
    }

    private func downloader() throws -> ModelDownloader {
        guard let url = URL(string: source), url.scheme != nil else {
            throw SpeechError.downloadFailed("'\(source)' is not a valid URL")
        }
        return try ModelDownloader(source: url)
    }

    /// Load without touching the network when the model is already cached.
    func loadIfCached() async {
        guard case .needsModel = state,
              let downloader = try? downloader(),
              await downloader.isAvailable(language: language, decoders: [decoder])
        else { return }
        await load()
    }

    func load() async {
        if let loaded, loaded == (language, decoder), recognizer != nil {
            state = .ready
            return
        }
        do {
            let downloader = try downloader()
            let language = language, decoder = decoder

            let didDownload = await !downloader.isAvailable(language: language,
                                                             decoders: [decoder])
            if didDownload {
                state = .downloading(file: "manifest.json", fraction: 0, index: 0, count: 1)
            }
            let directory = try await downloader.fetch(
                language: language, decoders: [decoder]
            ) { [weak self] progress in
                Task { @MainActor in
                    self?.state = .downloading(file: progress.file,
                                               fraction: progress.fraction,
                                               index: progress.fileIndex,
                                               count: progress.fileCount)
                }
            }

            state = .loading
            downloadedFromHost = didDownload ? URL(string: source)?.host : downloadedFromHost
            recognizer = try await Task.detached(priority: .userInitiated) {
                try SpeechRecognizer(modelsAt: directory, language: language,
                                     decoders: [decoder])
            }.value
            loaded = (language, decoder)
            cachedBytes = Self.directorySize(directory)
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func unload() {
        recognizer = nil
        loaded = nil
        state = .needsModel
    }

    func deleteCache() async {
        unload()
        if let downloader = try? downloader() {
            _ = try? await downloader.removeAll()
        }
        cachedBytes = 0
        outcomes.removeAll()
    }

    /// Start or stop continuous transcription.
    func toggleLive() async {
        if isLive {
            await live?.stop()
            await liveTask?.value
            live = nil
            liveTask = nil
            isLive = false
            liveStatus = ""
            livePartial = ""
            liveStable = ""
            return
        }

        if recognizer == nil { await load() }
        guard let recognizer else { return }

        var configuration = LiveTranscriber.Configuration()
        configuration.partialInterval = livePartialsEnabled ? 1.5 : 0
        let transcriber = LiveTranscriber(recognizer: recognizer,
                                          configuration: configuration)

        guard await transcriber.requestPermission() else {
            state = .failed("Microphone access was denied")
            return
        }

        do {
            let updates = try await transcriber.start()
            live = transcriber
            isLive = true
            liveTranscript = ""
            livePartial = ""
            liveStable = ""
            liveStatus = "Listening"

            liveTask = Task { [weak self] in
                for await update in updates {
                    guard let self else { return }
                    switch update {
                    case .speechStarted:
                        self.liveStatus = "Speaking"
                    case .speechEnded:
                        self.liveStatus = "Listening"
                        self.livePartial = ""
                        self.liveStable = ""
                    case .partial(let partial):
                        self.liveStable = partial.stablePrefix
                        self.livePartial = String(partial.text
                            .dropFirst(partial.stablePrefix.count))
                            .trimmingCharacters(in: .whitespaces)
                    case .final(let result):
                        self.liveTranscript += (self.liveTranscript.isEmpty ? "" : " ")
                            + result.text
                        self.livePartial = ""
                        self.liveStable = ""
                    case .failed(let message):
                        self.liveStatus = "Failed: \(message)"
                    }
                }
            }
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    /// Runs every bundled clip through both decoders and summarises the result.
    ///
    /// Loads its own recognizer with both decoders so the encoder is shared
    /// rather than paid for twice, and so the benchmark does not disturb
    /// whatever the main UI already has loaded.
    func runBenchmark() async {
        guard let base = URL(string: source) else {
            state = .failed("Invalid model URL"); return
        }
        let clips = Benchmark.clips
        guard !clips.isEmpty else {
            state = .failed("No benchmark clips bundled"); return
        }
        benchmarkRuns = []

        do {
            benchmarkProgress = "Fetching model"
            let downloader = try ModelDownloader(source: base)
            let directory = try await downloader.fetch(
                language: language, decoders: [.ctc, .rnnt]
            ) { [weak self] p in
                Task { @MainActor in
                    self?.benchmarkProgress =
                        "Downloading \(Int(p.totalFraction * 100))%"
                }
            }

            benchmarkProgress = "Loading both decoders"
            let lang = language
            let engine = try await Task.detached(priority: .userInitiated) {
                try SpeechRecognizer(modelsAt: directory, language: lang,
                                     decoders: [.ctc, .rnnt])
            }.value

            let total = clips.count * 2
            var done = 0
            for decoder in [SpeechRecognizer.Decoder.ctc, .rnnt] {
                var rows: [(audio: Double, processing: Double, wer: Double)] = []
                for clip in clips {
                    guard let url = clip.url else { continue }
                    done += 1
                    benchmarkProgress =
                        "\(decoder.rawValue.uppercased()) \(done)/\(total) — \(clip.file)"
                    let result = try await Task.detached(priority: .userInitiated) {
                        try engine.transcribe(contentsOf: url, using: decoder)
                    }.value
                    rows.append((clip.durationS,
                                 result.processingTime,
                                 errorRate(reference: clip.reference,
                                           hypothesis: result.text)))
                }
                benchmarkRuns.append(Benchmark.summarise(decoder: decoder, results: rows))
            }
            benchmarkProgress = nil
            state = .ready
        } catch {
            benchmarkProgress = nil
            state = .failed(error.localizedDescription)
        }
    }

    func transcribe(_ sample: Sample) async {
        guard let url = sample.url else {
            state = .failed("\(sample.file) is missing from the bundle")
            return
        }
        await run(label: sample.file, reference: sample.reference) {
            try $0.transcribe(contentsOf: url)
        }
    }

    func transcribe(fileAt url: URL) async {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let copy = FileManager.default.temporaryDirectory
            .appendingPathComponent(url.lastPathComponent)
        try? FileManager.default.removeItem(at: copy)
        do {
            try FileManager.default.copyItem(at: url, to: copy)
        } catch {
            state = .failed("Could not read \(url.lastPathComponent)")
            return
        }
        await run(label: url.lastPathComponent, reference: nil) {
            try $0.transcribe(contentsOf: copy)
        }
    }

    func toggleRecording() async {
        if isRecording {
            let captured = await microphone.stop()
            isRecording = false
            guard captured.count > 1_600 else {
                state = .failed("That recording was too short")
                return
            }
            await run(label: "Microphone", reference: nil) { try $0.transcribe(captured) }
            return
        }

        guard await microphone.requestPermission() else {
            state = .failed("Microphone access was denied")
            return
        }
        if recognizer == nil { await load() }
        guard recognizer != nil else { return }
        do {
            try await microphone.start()
            isRecording = true
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    private func run(label: String,
                     reference: String?,
                     body: @escaping @Sendable (SpeechRecognizer) throws
                        -> SpeechRecognizer.Transcription) async {
        if recognizer == nil { await load() }
        guard let recognizer else { return }

        state = .working
        do {
            let transcription = try await Task.detached(priority: .userInitiated) {
                try body(recognizer)
            }.value
            outcomes.insert(Outcome(
                source: label,
                text: transcription.text,
                language: transcription.language,
                decoder: transcription.decoder,
                realTimeFactor: transcription.realTimeFactor,
                duration: transcription.audioDuration,
                wordErrorRate: reference.map {
                    errorRate(reference: $0, hypothesis: transcription.text)
                }), at: 0)
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    private static func directorySize(_ url: URL) -> Int64 {
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: url.path)
        else { return 0 }
        return names.reduce(0) { total, name in
            let attributes = try? FileManager.default.attributesOfItem(
                atPath: url.appendingPathComponent(name).path)
            return total + ((attributes?[.size] as? Int64) ?? 0)
        }
    }
}
