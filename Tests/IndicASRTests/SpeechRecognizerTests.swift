//
//  SpeechRecognizerTests.swift
//  IndicASRTests
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import XCTest
@testable import IndicASR

/// End-to-end checks against `tools/run_reference.py` running the same int8
/// graphs. Skipped unless the quantized model is present; set
/// `INDICASR_MODEL_DIR` to override the default `models/int8_nc`.
final class SpeechRecognizerTests: XCTestCase {

    private struct Reference: Decodable {
        let reference: String
        let ctc: String
        let ctcWer: Double

        enum CodingKeys: String, CodingKey {
            case reference, ctc
            case ctcWer = "ctc_wer"
        }
    }

    /// Swift and numpy do not agree bit for bit: vDSP's FFT differs from numpy's
    /// by ~3.5e-05, which is enough to flip a near-tied argmax. Assert on the
    /// things that matter instead: the text barely moves, and WER does not get
    /// worse.
    func testCTCTracksPythonReference() throws {
        let recognizer = try SpeechRecognizer(modelsAt: try Fixtures.modelDirectory(), language: .hindi)
        var identical = 0

        for (file, expected) in try Fixtures.decode([String: Reference].self, from: "python_reference.json").sorted(by: { $0.key < $1.key }) {
            let result = try recognizer.transcribe(contentsOf: try Fixtures.url(file))
            let drift = editDistance(result.text, expected.ctc)
            let wer = wordErrorRate(reference: expected.reference, hypothesis: result.text)
            if result.text == expected.ctc { identical += 1 }

            XCTAssertLessThan(drift, 0.02, "\(file) drifted \(drift) from the reference")
            XCTAssertLessThanOrEqual(wer, expected.ctcWer + 0.02,
                                     "\(file) WER \(wer) is worse than Python's \(expected.ctcWer)")
            print(String(format: "  %@ rtf %.3f drift %.4f wer %.4f (python %.4f)",
                         file, result.realTimeFactor, drift, wer, expected.ctcWer))
        }
        print("  byte-identical on \(identical) of \(try Fixtures.decode([String: Reference].self, from: "python_reference.json").count)")
    }

    /// The transducer has its own index-space traps, so it gets its own check.
    /// Feeding global rather than local token ids yields WER above 3.
    func testTransducerProducesUsableText() throws {
        guard let (file, expected) = try Fixtures.decode([String: Reference].self, from: "python_reference.json").sorted(by: { $0.key < $1.key }).first
        else { throw XCTSkip("no references bundled") }

        let recognizer = try SpeechRecognizer(modelsAt: try Fixtures.modelDirectory(),
                                              language: .hindi,
                                              decoders: [.rnnt])
        let result = try recognizer.transcribe(contentsOf: try Fixtures.url(file))

        XCTAssertFalse(result.text.isEmpty)
        let wer = wordErrorRate(reference: expected.reference, hypothesis: result.text)
        XCTAssertLessThan(wer, 0.35, "transducer WER \(wer); check SOS and token feedback")
        print(String(format: "  rnnt %@ wer %.4f rtf %.3f", file, wer, result.realTimeFactor))
    }

    func testAskingForAnUnloadedDecoderThrows() throws {
        let recognizer = try SpeechRecognizer(modelsAt: try Fixtures.modelDirectory(), language: .hindi,
                                              decoders: [.ctc])
        XCTAssertThrowsError(try recognizer.transcribe([0.1, 0.2], using: .rnnt))
    }

    func testEmptyAudioThrows() throws {
        let recognizer = try SpeechRecognizer(modelsAt: try Fixtures.modelDirectory(), language: .hindi)
        XCTAssertThrowsError(try recognizer.transcribe([])) { error in
            guard case SpeechError.emptyAudio = error else {
                return XCTFail("expected .emptyAudio, got \(error)")
            }
        }
    }

    func testVocabularyLayout() throws {
        let directory = try Fixtures.modelDirectory()
        for language in [Language.hindi, .tamil, .assamese, .urdu] {
            let vocabulary = try Vocabulary(language: language, assetsDirectory: directory)
            XCTAssertEqual(vocabulary.tokens.count, 257, language.code)
            XCTAssertEqual(vocabulary.maskIndices.count, 257, language.code)
            XCTAssertEqual(vocabulary.tokens[Vocabulary.blank], "|", language.code)
            XCTAssertEqual(vocabulary.maskIndices.first, Int32(language.vocabularyOffset))
            XCTAssertEqual(vocabulary.maskIndices.last, Int32(Vocabulary.startOfSequence))
        }
    }

    func testEveryLanguageHasAVocabulary() throws {
        let directory = try Fixtures.modelDirectory()
        for language in Language.allCases {
            XCTAssertNoThrow(try Vocabulary(language: language, assetsDirectory: directory),
                             "missing vocabulary for \(language.code)")
        }
    }
}

private func editDistance(_ a: String, _ b: String) -> Double {
    levenshtein(Array(a).map(String.init), Array(b).map(String.init))
        / Double(max(a.count, b.count, 1))
}

private func wordErrorRate(reference: String, hypothesis: String) -> Double {
    let words = reference.split(separator: " ").map(String.init)
    guard !words.isEmpty else { return hypothesis.isEmpty ? 0 : 1 }
    return levenshtein(words, hypothesis.split(separator: " ").map(String.init))
        / Double(words.count)
}

private func levenshtein(_ a: [String], _ b: [String]) -> Double {
    if a.isEmpty { return Double(b.count) }
    if b.isEmpty { return Double(a.count) }
    var previous = Array(0...b.count)
    var current = [Int](repeating: 0, count: b.count + 1)
    for i in 1...a.count {
        current[0] = i
        for j in 1...b.count {
            current[j] = min(previous[j] + 1, current[j - 1] + 1,
                             previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1))
        }
        swap(&previous, &current)
    }
    return Double(previous[b.count])
}
