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

    @Published var language: Language = .hindi { didSet { invalidate(oldValue != language) } }
    @Published var decoder: SpeechRecognizer.Decoder = .ctc {
        didSet { invalidate(oldValue != decoder) }
    }
    /// Default host for the demo. Override it in the app's Settings section, or
    /// point it at `tools/dev.sh serve` when testing changes to the model.
    @AppStorage("modelSource") var source = RecognizerModel.defaultSource
    static let defaultSource = "https://pub-2b0e4d9b945b42f580f49f58405101d7.r2.dev"

    let samples = Sample.bundled

    @Published private var downloadedFromHost: String?
    private var recognizer: SpeechRecognizer?
    private var loaded: (Language, SpeechRecognizer.Decoder)?
    private let microphone = MicrophoneCapture()

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
