//
//  SpeechRecognizer.swift
//  IndicASR
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import Accelerate
import Foundation
import OnnxRuntimeBindings

/// Offline speech-to-text for 22 Indian languages, using AI4Bharat's
/// IndicConformer 600M through ONNX Runtime.
///
///     let asr = try SpeechRecognizer(modelsAt: directory, language: .hindi)
///     let result = try asr.transcribe(contentsOf: wavURL)
///     print(result.text)
///
/// Loading pulls roughly 900 MB of weights into memory, and `transcribe` is
/// CPU-bound and blocking. Call both off the main thread.
///
/// The pipeline mirrors AI4Bharat's reference implementation:
///
///     samples -> log-mel (80 bins)
///             -> encoder            (1,80,T) -> (1,1024,T/8)
///             -> ctc_decoder        -> (1,T/8,5633) -> language mask -> greedy
///             or joint_enc + rnnt_decoder + joint_pred + joint_post_net_<lang>
///
/// The encoder subsamples by 8, so one output frame covers 80 ms.
public final class SpeechRecognizer: @unchecked Sendable {

    public enum Decoder: String, Sendable, CaseIterable, Identifiable {
        /// Greedy CTC. Faster, and what you want unless you have a reason.
        case ctc
        /// Greedy transducer. Slightly better on some audio, several times slower.
        case rnnt

        public var id: String { rawValue }
    }

    public struct Transcription: Sendable {
        public let text: String
        public let language: Language
        public let decoder: Decoder
        public let audioDuration: TimeInterval
        public let processingTime: TimeInterval

        public var realTimeFactor: Double {
            audioDuration > 0 ? processingTime / audioDuration : 0
        }
    }

    public let language: Language
    public let decoders: Set<Decoder>

    private static let encoderDimension = 1024
    private static let predictionLayers = 2
    private static let predictionHidden = 640
    private static let maxSymbolsPerFrame = 10

    private let environment: ORTEnv
    private let mel: MelSpectrogram
    private let vocabulary: Vocabulary
    private let encoder: ORTSession
    private let ctc: ORTSession?
    private let transducer: Transducer?

    private struct Transducer {
        let decoder: ORTSession
        let jointEncoder: ORTSession
        let jointPrediction: ORTSession
        let jointPreNet: ORTSession
        let jointPostNet: ORTSession
    }

    /// - Parameters:
    ///   - directory: folder holding the `.onnx` files, `vocab.json` and
    ///     `language_masks.json`. Both the upstream `assets/` layout and a flat
    ///     quantized output directory work.
    ///   - decoders: which decoders to load. `[.ctc]` skips about 40 MB of RNNT
    ///     graphs. Loading both shares the one encoder rather than paying for
    ///     it twice.
    ///   - threads: intra-op threads; 0 lets ONNX Runtime decide.
    public init(modelsAt directory: URL,
                language: Language,
                decoders: Set<Decoder> = [.ctc],
                threads: Int32 = 0) throws {
        precondition(!decoders.isEmpty, "at least one decoder is required")

        // Locals throughout: a nested helper touching `self` before every stored
        // property is initialized will not compile.
        let env = try ORTEnv(loggingLevel: .warning)
        let assets = Self.assetsDirectory(in: directory)
        func graph(_ name: String) throws -> ORTSession {
            try ONNX.session(env: env,
                             path: assets.appendingPathComponent("\(name).onnx"),
                             threads: threads)
        }

        self.language = language
        self.decoders = decoders
        environment = env
        mel = MelSpectrogram()
        vocabulary = try Vocabulary(language: language, assetsDirectory: assets)
        encoder = try graph("encoder")
        ctc = decoders.contains(.ctc) ? try graph("ctc_decoder") : nil
        transducer = decoders.contains(.rnnt)
            ? Transducer(decoder: try graph("rnnt_decoder"),
                         jointEncoder: try graph("joint_enc"),
                         jointPrediction: try graph("joint_pred"),
                         jointPreNet: try graph("joint_pre_net"),
                         jointPostNet: try graph("joint_post_net_\(language.code)"))
            : nil
    }

    /// Transcribe mono samples in [-1, 1] at 16 kHz.
    public func transcribe(_ samples: [Float],
                           using decoder: Decoder? = nil) throws -> Transcription {
        guard !samples.isEmpty else { throw SpeechError.emptyAudio }
        let choice = try resolve(decoder)

        let started = Date()
        let (encoded, frames) = try encode(samples)
        let text = switch choice {
        case .ctc:  try decodeCTC(encoded, frames: frames)
        case .rnnt: try decodeTransducer(encoded, frames: frames)
        }

        return Transcription(text: text,
                             language: language,
                             decoder: choice,
                             audioDuration: Double(samples.count) / 16_000,
                             processingTime: Date().timeIntervalSince(started))
    }

    /// Transcribe any file AVFoundation can decode; resampled to 16 kHz mono.
    public func transcribe(contentsOf url: URL,
                           using decoder: Decoder? = nil) throws -> Transcription {
        try transcribe(AudioFile.samples(at: url), using: decoder)
    }

    private func resolve(_ requested: Decoder?) throws -> Decoder {
        let choice = requested ?? (decoders.contains(.ctc) ? .ctc : .rnnt)
        guard decoders.contains(choice) else {
            throw SpeechError.missingModelFile(
                "\(choice.rawValue) decoder was not loaded; pass it to init")
        }
        return choice
    }

    private static func assetsDirectory(in directory: URL) -> URL {
        let nested = directory.appendingPathComponent("assets")
        let marker = nested.appendingPathComponent("encoder.onnx")
        return FileManager.default.fileExists(atPath: marker.path) ? nested : directory
    }

    /// - Returns: encoder output as row-major (1024, frames), and the frame count.
    private func encode(_ samples: [Float]) throws -> ([Float], Int) {
        let (features, frames) = mel(samples)
        guard frames > 0 else { throw SpeechError.emptyAudio }

        let outputs = try ONNX.run(
            encoder,
            inputs: ["audio_signal": try ONNX.tensor(features, shape: [1, 80, frames]),
                     "length": try ONNX.tensor([Int64(frames)], shape: [1])],
            outputs: ["outputs", "encoded_lengths"])

        let shape = try ONNX.shape(of: outputs[0])
        guard shape.count == 3, shape[1] == Self.encoderDimension else {
            throw SpeechError.shapeMismatch(
                "encoder produced \(shape), expected [1, 1024, T]")
        }
        let length = try ONNX.int64s(outputs[1]).first.map(Int.init) ?? shape[2]
        return (try ONNX.floats(outputs[0]), min(length, shape[2]))
    }

    private func decodeCTC(_ encoded: [Float], frames: Int) throws -> String {
        guard let ctc else { throw SpeechError.missingModelFile("ctc_decoder.onnx") }

        let total = encoded.count / Self.encoderDimension
        let logits = try ONNX.run(
            ctc,
            inputs: ["encoder_output": try ONNX.tensor(
                encoded, shape: [1, Self.encoderDimension, total])],
            outputs: ["logprobs"])[0]

        let shape = try ONNX.shape(of: logits)
        guard shape.count == 3, shape[2] == Vocabulary.globalSize else {
            throw SpeechError.shapeMismatch(
                "ctc_decoder produced \(shape), expected [1, T, \(Vocabulary.globalSize)]")
        }
        let values = try ONNX.floats(logits)

        // argmax over the language's 257 columns. Softmax is order preserving,
        // so the reference's log_softmax before argmax would be wasted work.
        var best = [Int]()
        best.reserveCapacity(min(frames, shape[1]))
        for t in 0..<min(frames, shape[1]) {
            let row = t * Vocabulary.globalSize
            var index = 0
            var value = -Float.greatestFiniteMagnitude
            for (local, global) in vocabulary.maskIndices.enumerated()
            where values[row + Int(global)] > value {
                value = values[row + Int(global)]
                index = local
            }
            best.append(index)
        }

        let collapsed = best.enumerated().compactMap { i, id in
            i == 0 || id != best[i - 1] ? id : nil
        }
        return vocabulary.text(from: collapsed)
    }

    /// Frame-synchronous greedy transducer search. The loop lives here because
    /// ONNX cannot express it; ORT runs one step per symbol.
    private func decodeTransducer(_ encoded: [Float], frames: Int) throws -> String {
        guard let transducer else { throw SpeechError.missingModelFile("rnnt_decoder.onnx") }

        let total = encoded.count / Self.encoderDimension
        var timeMajor = [Float](repeating: 0, count: encoded.count)
        for d in 0..<Self.encoderDimension {
            let row = d * total
            for t in 0..<total { timeMajor[t * Self.encoderDimension + d] = encoded[row + t] }
        }

        let projected = try ONNX.floats(try ONNX.run(
            transducer.jointEncoder,
            inputs: ["input": try ONNX.tensor(
                timeMajor, shape: [1, total, Self.encoderDimension])],
            outputs: ["output"])[0])

        let hidden = Self.predictionHidden
        var hypothesis = [Int]()
        var previous = Int32(Vocabulary.startOfSequence)
        var h = [Float](repeating: 0, count: Self.predictionLayers * hidden)
        var c = h

        for t in 0..<min(frames, total) {
            let frame = Array(projected[(t * hidden)..<((t + 1) * hidden)])

            for _ in 0..<Self.maxSymbolsPerFrame {
                let state = try ONNX.run(
                    transducer.decoder,
                    inputs: [
                        "targets": try ONNX.tensor([previous], shape: [1, 1]),
                        "target_length": try ONNX.tensor([Int32(1)], shape: [1]),
                        "states.1": try ONNX.tensor(h, shape: [Self.predictionLayers, 1, hidden]),
                        "onnx::Slice_3": try ONNX.tensor(c, shape: [Self.predictionLayers, 1, hidden]),
                    ],
                    outputs: ["outputs", "states", "162"])

                // The decoder emits (1, 640, 1) and joint_pred wants (1, 1, 640);
                // with one symbol that transpose is a reshape, so no copy.
                let prediction = try ONNX.floats(
                    try ONNX.run(transducer.jointPrediction,
                                 inputs: ["input": try ONNX.tensor(
                                     try ONNX.floats(state[0]), shape: [1, 1, hidden])],
                                 outputs: ["output"])[0])

                var joint = [Float](repeating: 0, count: hidden)
                vDSP_vadd(frame, 1, prediction, 1, &joint, 1, vDSP_Length(hidden))

                let activated = try ONNX.run(
                    transducer.jointPreNet,
                    inputs: ["input": try ONNX.tensor(joint, shape: [1, 1, hidden])],
                    outputs: ["output"])[0]
                let logits = try ONNX.floats(
                    try ONNX.run(transducer.jointPostNet,
                                 inputs: ["input": activated],
                                 outputs: ["output"])[0])

                var token = 0
                var value = -Float.greatestFiniteMagnitude
                for (i, v) in logits.enumerated() where v > value { value = v; token = i }
                if token == Vocabulary.blank { break }

                hypothesis.append(token)
                // Local, per-language ids feed straight back, even though the
                // embedding is (5633, 640). Remapping to global ids collapses
                // decoding to repeated garbage.
                previous = Int32(token)
                h = try ONNX.floats(state[1])
                c = try ONNX.floats(state[2])
            }
        }
        return vocabulary.text(from: hypothesis)
    }
}
