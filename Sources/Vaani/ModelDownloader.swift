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
        /// Bytes received across the whole download so far.
        public let totalBytesReceived: Int64
        /// Total size of everything being downloaded.
        public let totalBytesExpected: Int64

        /// Progress through the current file, 0 to 1.
        public var fraction: Double {
            bytesExpected > 0 ? Double(bytesReceived) / Double(bytesExpected) : 0
        }

        /// Progress across the whole download, 0 to 1. Prefer this for a
        /// progress bar: the encoder is about 97% of the bytes, so per-file
        /// progress spends almost all its time on one file.
        public var totalFraction: Double {
            totalBytesExpected > 0
                ? Double(totalBytesReceived) / Double(totalBytesExpected) : 0
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
        /// Language code to that language's files. Each carries its own
        /// checksum: the per-language heads are all the same size, so a shared
        /// entry would silently serve one language's head for another.
        let perLanguage: [String: [String: Entry]]
        let profiles: [String: [String]]
    }

    /// Where model files are cached. Pass this to
    /// ``SpeechRecognizer/init(modelsAt:language:decoders:threads:)``.
    public let directory: URL

    private let source: URL
    private let session: URLSession
    private let delegate = DownloadDelegate()
    private var cached: Manifest?
    private var completedBytes: Int64 = 0
    private var totalBytes: Int64 = 0
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
        // A delegate, not downloadTask(with:completionHandler:) plus KVO.
        // Progress.completedUnitCount is coalesced and fired twice for a 23 MB
        // transfer in testing, which leaves a progress bar apparently frozen for
        // the 878 MB encoder. didWriteData reports every chunk.
        session = URLSession(configuration: configuration,
                             delegate: delegate,
                             delegateQueue: nil)

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

        totalBytes = missing.reduce(0) { $0 + $1.entry.size }
        completedBytes = 0
        for (index, file) in missing.enumerated() {
            try await download(file, index: index, of: missing.count, onProgress: onProgress)
            completedBytes += file.entry.size
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
            guard let files = manifest.perLanguage[language.code] else {
                throw SpeechError.malformedModel(
                    "manifest has no files for language '\(language.code)'")
            }
            required.append(contentsOf: files.sorted { $0.key < $1.key }
                .map { (name: $0.key, entry: $0.value) })
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
        let staged = directory.appendingPathComponent(file.name + ".part")
        try? FileManager.default.removeItem(at: staged)

        let expected = file.entry.size
        let alreadyDone = completedBytes
        let overall = totalBytes
        let report: @Sendable (Int64, Int64) -> Void = { received, fileTotal in
            onProgress?(Progress(file: file.name,
                                 fileIndex: index,
                                 fileCount: count,
                                 bytesReceived: received,
                                 bytesExpected: fileTotal > 0 ? fileTotal : expected,
                                 totalBytesReceived: alreadyDone + received,
                                 totalBytesExpected: overall))
        }
        report(0, expected)

        let task = session.downloadTask(with: source.appendingPathComponent(file.name))
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            delegate.register(task, movingTo: staged, name: file.name,
                              progress: report) { result in
                continuation.resume(with: result)
            }
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
        report(expected, expected)
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

/// Bridges `URLSessionDownloadDelegate` to async/await.
///
/// The temporary file is deleted as soon as `didFinishDownloadingTo` returns,
/// so the move happens inside that callback rather than afterwards.
private final class DownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {

    private struct Pending {
        let destination: URL
        let name: String
        let progress: @Sendable (Int64, Int64) -> Void
        let finish: @Sendable (Result<Void, Error>) -> Void
        var moved: Result<Void, Error>?
    }

    private let lock = NSLock()
    private var pending: [Int: Pending] = [:]

    func register(_ task: URLSessionTask,
                  movingTo destination: URL,
                  name: String,
                  progress: @escaping @Sendable (Int64, Int64) -> Void,
                  finish: @escaping @Sendable (Result<Void, Error>) -> Void) {
        lock.withLock {
            pending[task.taskIdentifier] = Pending(destination: destination, name: name,
                                                   progress: progress, finish: finish)
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        let handler = lock.withLock { pending[downloadTask.taskIdentifier]?.progress }
        handler?(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        guard var entry = lock.withLock({ pending[downloadTask.taskIdentifier] }) else { return }

        let code = (downloadTask.response as? HTTPURLResponse)?.statusCode ?? -1
        if code != 200 {
            entry.moved = .failure(SpeechError.downloadFailed(
                "\(entry.name) returned HTTP \(code)"))
        } else {
            do {
                try? FileManager.default.removeItem(at: entry.destination)
                try FileManager.default.moveItem(at: location, to: entry.destination)
                entry.moved = .success(())
            } catch {
                entry.moved = .failure(SpeechError.downloadFailed(
                    "\(entry.name): could not stage the download"))
            }
        }
        lock.withLock { pending[downloadTask.taskIdentifier] = entry }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        guard let entry = lock.withLock({ pending.removeValue(forKey: task.taskIdentifier) })
        else { return }

        if let error {
            entry.finish(.failure(SpeechError.downloadFailed(
                "\(entry.name): \(error.localizedDescription)")))
        } else {
            entry.finish(entry.moved ?? .failure(SpeechError.downloadFailed(
                "\(entry.name): finished without producing a file")))
        }
    }
}
