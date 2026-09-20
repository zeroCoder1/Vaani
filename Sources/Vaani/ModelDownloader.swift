//
//  ModelDownloader.swift
//  Vaani
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import CryptoKit
import Foundation

/// Fetches model files on demand and caches them on disk.
///
/// The encoder is ~870 MB and shared by every language; a per-language RNNT
/// head is ~0.17 MB. Adding a second language after the first is therefore
/// nearly free, and `fetch` only downloads what is actually missing.
///
/// Offline-first: if the cached manifest says everything is present, `fetch`
/// returns without touching the network. Side-loading the files plus
/// `manifest.json` into `directory` works for the same reason.
///
/// Generate `manifest.json` with `tools/make_manifest.py` and host the
/// directory anywhere static.
public actor ModelDownloader {

    public struct Progress: Sendable {
        public let file: String
        /// Zero-based index of this file within the current download.
        public let fileIndex: Int
        /// How many files this download covers in total.
        public let fileCount: Int
        /// Bytes written so far for this file.
        public let bytesReceived: Int64
        /// Expected size of this file, from the manifest or the response.
        public let bytesExpected: Int64

        /// Progress through the current file, 0 to 1. This is per file, not
        /// across the whole download; the encoder dominates by size, so
        /// weight by `bytesExpected` for an overall figure.
        public var fraction: Double {
            bytesExpected > 0 ? Double(bytesReceived) / Double(bytesExpected) : 0
        }
    }

    struct Manifest: Codable, Sendable {
        struct Entry: Codable, Sendable {
            let size: Int64
            let sha256: String?
        }
        let version: String
        let variant: String
        let shared: [String: Entry]
        /// Keys contain `{lang}`, substituted per request.
        let perLanguage: [String: Entry]
        let profiles: [String: [String]]
    }

    /// Where model files are cached. Pass this to
    /// ``SpeechRecognizer/init(modelsAt:language:decoders:threads:)``.
    public let directory: URL

    private let source: URL
    private let session: URLSession
    private var cached: Manifest?
    private var observations: [NSKeyValueObservation] = []

    /// - Parameters:
    ///   - source: directory URL containing `manifest.json` and the model files.
    ///   - cacheDirectory: defaults to Application Support/Vaani.
    public init(source: URL, cacheDirectory: URL? = nil) throws {
        self.source = source
        directory = try cacheDirectory ?? FileManager.default
            .url(for: .applicationSupportDirectory, in: .userDomainMask,
                 appropriateFor: nil, create: true)
            .appendingPathComponent("Vaani", isDirectory: true)

        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForResource = 3_600   // ~900 MB on cellular
        configuration.timeoutIntervalForRequest = 60       // stalled, not slow
        // Deliberately false. When true, iOS silently waits instead of failing
        // if local-network access has not been granted, which looks exactly
        // like a hang with no error for up to timeoutIntervalForResource.
        configuration.waitsForConnectivity = false
        session = URLSession(configuration: configuration)

        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var url = directory
        var values = URLResourceValues()
        values.isExcludedFromBackup = true   // everything here is re-downloadable
        try? url.setResourceValues(values)
    }

    /// Whether everything needed is already on disk. Never hits the network.
    public func isAvailable(language: Language,
                            decoders: Set<SpeechRecognizer.Decoder> = [.ctc]) -> Bool {
        guard let manifest = storedManifest(),
              let required = try? files(for: language, decoders: decoders, in: manifest)
        else { return false }
        return required.allSatisfy(isPresent)
    }

    /// Download whatever is missing and return the directory to hand to
    /// `SpeechRecognizer(modelsAt:)`.
    @discardableResult
    public func fetch(language: Language,
                      decoders: Set<SpeechRecognizer.Decoder> = [.ctc],
                      onProgress: (@Sendable (Progress) -> Void)? = nil) async throws -> URL {
        if let manifest = storedManifest(),
           let required = try? files(for: language, decoders: decoders, in: manifest),
           required.allSatisfy(isPresent) {
            cached = manifest
            return directory
        }

        let manifest = try await remoteManifest()
        let missing = try files(for: language, decoders: decoders, in: manifest)
            .filter { !isPresent($0) }

        for (index, file) in missing.enumerated() {
            try await download(file, index: index, of: missing.count, onProgress: onProgress)
        }
        store(manifest)
        return directory
    }

    /// Delete the cache. Returns bytes reclaimed.
    @discardableResult
    public func removeAll() throws -> Int64 {
        let manager = FileManager.default
        var freed: Int64 = 0
        for name in try manager.contentsOfDirectory(atPath: directory.path) {
            let url = directory.appendingPathComponent(name)
            freed += (try? manager.attributesOfItem(atPath: url.path))
                .flatMap { $0[.size] as? Int64 } ?? 0
            try? manager.removeItem(at: url)
        }
        cached = nil
        return freed
    }

    private typealias File = (name: String, entry: Manifest.Entry)

    private func files(for language: Language,
                       decoders: Set<SpeechRecognizer.Decoder>,
                       in manifest: Manifest) throws -> [File] {
        var names = Set<String>()
        for decoder in decoders {
            guard let listed = manifest.profiles[decoder.rawValue] else {
                throw SpeechError.malformedModel(
                    "manifest has no '\(decoder.rawValue)' profile")
            }
            names.formUnion(listed)
        }

        var required: [File] = try names.sorted().map {
            guard let entry = manifest.shared[$0] else {
                throw SpeechError.malformedModel("manifest lists unknown file '\($0)'")
            }
            return ($0, entry)
        }
        if decoders.contains(.rnnt) {
            for (pattern, entry) in manifest.perLanguage {
                required.append((pattern.replacingOccurrences(of: "{lang}", with: language.code),
                                 entry))
            }
        }
        return required
    }

    private func isPresent(_ file: File) -> Bool {
        let url = directory.appendingPathComponent(file.name)
        guard let size = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.size]
                as? Int64 else { return false }
        return size == file.entry.size
    }

    private var storedManifestURL: URL { directory.appendingPathComponent("manifest.json") }

    private func storedManifest() -> Manifest? {
        guard let data = try? Data(contentsOf: storedManifestURL) else { return nil }
        return try? JSONDecoder().decode(Manifest.self, from: data)
    }

    private func store(_ manifest: Manifest) {
        guard let data = try? JSONEncoder().encode(manifest) else { return }
        try? data.write(to: storedManifestURL, options: .atomic)
    }

    private func remoteManifest() async throws -> Manifest {
        if let cached { return cached }
        let (data, response) = try await session.data(
            from: source.appendingPathComponent("manifest.json"))
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw SpeechError.downloadFailed("manifest.json returned HTTP \(code)")
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        cached = manifest
        return manifest
    }

    private func download(_ file: File, index: Int, of count: Int,
                          onProgress: (@Sendable (Progress) -> Void)?) async throws {
        let destination = directory.appendingPathComponent(file.name)
        let remote = source.appendingPathComponent(file.name)
        let staged = directory.appendingPathComponent(file.name + ".part")
        try? FileManager.default.removeItem(at: staged)

        // Report before any bytes arrive so the UI names the file it is on
        // rather than sitting on whatever was shown last.
        let expected = file.entry.size
        onProgress?(Progress(file: file.name, fileIndex: index, fileCount: count,
                             bytesReceived: 0, bytesExpected: expected))

        // downloadTask with a completion handler and KVO progress, rather than
        // download(from:delegate:). A URLSessionDownloadDelegate has to
        // implement didFinishDownloadingTo, and that competes with the async
        // variant's own completion handling - on iOS the continuation can fail
        // to resume, which presents as an unbreakable hang with no error.
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let task = session.downloadTask(with: remote) { temporary, response, error in
                if let error {
                    continuation.resume(throwing: SpeechError.downloadFailed(
                        "\(file.name): \(error.localizedDescription)"))
                    return
                }
                let code = (response as? HTTPURLResponse)?.statusCode ?? -1
                guard code == 200, let temporary else {
                    continuation.resume(throwing: SpeechError.downloadFailed(
                        "\(file.name) returned HTTP \(code)"))
                    return
                }
                // The temporary file is removed as soon as this handler returns.
                do {
                    try FileManager.default.moveItem(at: temporary, to: staged)
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: SpeechError.downloadFailed(
                        "\(file.name): could not stage the download"))
                }
            }

            let observation = task.progress.observe(\.completedUnitCount) { progress, _ in
                onProgress?(Progress(file: file.name, fileIndex: index, fileCount: count,
                                     bytesReceived: progress.completedUnitCount,
                                     bytesExpected: progress.totalUnitCount > 0
                                        ? progress.totalUnitCount : expected))
            }
            observations.append(observation)
            task.resume()
        }

        if let expectedHash = file.entry.sha256 {
            guard try Self.sha256(of: staged) == expectedHash else {
                try? FileManager.default.removeItem(at: staged)
                throw SpeechError.checksumMismatch(file: file.name)
            }
        }

        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: staged, to: destination)
        onProgress?(Progress(file: file.name, fileIndex: index, fileCount: count,
                             bytesReceived: expected, bytesExpected: expected))
    }

    /// Hash in chunks; these files do not fit comfortably in memory.
    private static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 4 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
