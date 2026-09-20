//
//  MicrophoneCapture.swift
//  Vaani
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

#if os(iOS)
import AVFoundation
import Foundation

/// Records microphone audio as mono 16 kHz samples, ready for `SpeechRecognizer`.
///
///     let mic = MicrophoneCapture()
///     guard await mic.requestPermission() else { return }
///     try await mic.start()
///     // ...
///     let result = try recognizer.transcribe(await mic.stop())
///
/// `start` and `stop` are async because `AVAudioSession.setCategory` and
/// `setActive` block, and calling them on the main thread makes the UI
/// unresponsive. They run on a private serial queue, which also means start and
/// stop can never interleave.
///
/// Add `NSMicrophoneUsageDescription` to your Info.plist or the app is
/// terminated on first use.
public final class MicrophoneCapture: @unchecked Sendable {

    private let engine = AVAudioEngine()
    private let queue = DispatchQueue(label: "dev.vaani.microphone")
    private let lock = NSLock()
    private var samples: [Float] = []
    private var tapInstalled = false
    private var handler: (@Sendable ([Float]) -> Void)?
    private var retains = true

    /// Creates a recorder. No audio session is touched until ``start()``.
    public init() {}

    /// Receive audio as it arrives, instead of waiting for ``stop()``.
    ///
    /// The handler is called on an internal audio queue with mono 16 kHz
    /// samples, so it must return quickly; hand work to another queue rather
    /// than doing it here.
    ///
    /// Set `retainingSamples` to false for long sessions. Retained audio grows
    /// at about 64 KB per second, so an hour of recording is roughly 230 MB,
    /// and ``stop()`` then returns nothing.
    public func stream(retainingSamples: Bool = false,
                       to handler: @escaping @Sendable ([Float]) -> Void) {
        lock.withLock {
            self.handler = handler
            self.retains = retainingSamples
        }
    }

    deinit {
        if tapInstalled { engine.inputNode.removeTap(onBus: 0) }
    }

    /// Whether the engine is currently capturing.
    public var isRecording: Bool { engine.isRunning }

    /// Seconds of audio captured so far. Safe to poll while recording.
    public var recordedDuration: TimeInterval {
        lock.withLock { Double(samples.count) / AudioFile.sampleRate }
    }

    /// Asks for microphone access, returning the granted state.
    ///
    /// Requires `NSMicrophoneUsageDescription` in your Info.plist; without it
    /// the app is terminated rather than shown a prompt.
    public func requestPermission() async -> Bool {
        await withCheckedContinuation { continuation in
            if #available(iOS 17.0, *) {
                AVAudioApplication.requestRecordPermission {
                    continuation.resume(returning: $0)
                }
            } else {
                AVAudioSession.sharedInstance().requestRecordPermission {
                    continuation.resume(returning: $0)
                }
            }
        }
    }

    /// Idempotent: starting while already recording does nothing.
    public func start() async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            queue.async {
                do {
                    try self.beginRecording()
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Stop and return everything captured. Safe to call when idle.
    @discardableResult
    public func stop() async -> [Float] {
        await withCheckedContinuation { (continuation: CheckedContinuation<[Float], Never>) in
            queue.async { continuation.resume(returning: self.endRecording()) }
        }
    }

    /// Discard anything captured without returning it.
    public func reset() async {
        _ = await stop()
        lock.withLock { samples.removeAll(keepingCapacity: false) }
    }

    // MARK: - serial queue

    private func beginRecording() throws {
        guard !engine.isRunning else { return }

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement, options: [.duckOthers])
        try session.setActive(true)

        // The hardware format is only meaningful once the session is active.
        let input = engine.inputNode
        let hardware = input.outputFormat(forBus: 0)
        guard hardware.sampleRate > 0, hardware.channelCount > 0 else {
            try? session.setActive(false, options: .notifyOthersOnDeactivation)
            throw SpeechError.unsupportedAudioFormat(
                "microphone reported an invalid format; is another app holding the input?")
        }

        guard let target = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                         sampleRate: AudioFile.sampleRate,
                                         channels: 1,
                                         interleaved: false),
              let converter = AVAudioConverter(from: hardware, to: target) else {
            try? session.setActive(false, options: .notifyOthersOnDeactivation)
            throw SpeechError.unsupportedAudioFormat(
                "cannot convert \(Int(hardware.sampleRate))Hz/\(hardware.channelCount)ch to 16kHz mono")
        }
        lock.withLock { samples.removeAll(keepingCapacity: true) }

        // installTap raises an Objective-C exception - uncatchable from Swift -
        // if the bus already has a tap, and a failed start can leave one behind.
        clearTap()
        input.installTap(onBus: 0, bufferSize: 4096, format: hardware) { [weak self] buffer, _ in
            self?.append(buffer, using: converter, target: target, from: hardware)
        }
        tapInstalled = true

        engine.prepare()
        do {
            try engine.start()
        } catch {
            // Unwind, or the orphaned tap crashes the next start.
            clearTap()
            try? session.setActive(false, options: .notifyOthersOnDeactivation)
            throw error
        }
    }

    private func endRecording() -> [Float] {
        clearTap()
        if engine.isRunning { engine.stop() }
        try? AVAudioSession.sharedInstance().setActive(
            false, options: .notifyOthersOnDeactivation)
        return lock.withLock { samples }
    }

    private func clearTap() {
        guard tapInstalled else { return }
        engine.inputNode.removeTap(onBus: 0)
        tapInstalled = false
    }

    private func append(_ buffer: AVAudioPCMBuffer,
                        using converter: AVAudioConverter,
                        target: AVAudioFormat,
                        from hardware: AVAudioFormat) {
        let capacity = AVAudioFrameCount(
            Double(buffer.frameLength) * AudioFile.sampleRate / hardware.sampleRate) + 1024
        guard let converted = AVAudioPCMBuffer(pcmFormat: target,
                                               frameCapacity: capacity) else { return }

        var consumed = false
        var failure: NSError?
        converter.convert(to: converted, error: &failure) { _, status in
            if consumed {
                status.pointee = .noDataNow
                return nil
            }
            consumed = true
            status.pointee = .haveData
            return buffer
        }
        guard failure == nil, let channel = converted.floatChannelData else { return }
        let chunk = Array(UnsafeBufferPointer(start: channel[0],
                                              count: Int(converted.frameLength)))

        let deliver: (@Sendable ([Float]) -> Void)? = lock.withLock {
            if retains { samples.append(contentsOf: chunk) }
            return handler
        }
        deliver?(chunk)
    }
}
#endif
