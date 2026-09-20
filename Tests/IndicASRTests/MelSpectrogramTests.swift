//
//  MelSpectrogramTests.swift
//  IndicASRTests
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import XCTest
@testable import IndicASR

/// Golden vectors come from `tools/reference_frontend.py`, which is itself
/// cross-validated against librosa and AI4Bharat's TorchScript preprocessor.
final class MelSpectrogramTests: XCTestCase {

    func testFilterbankMatchesLibrosa() throws {
        let expected = try Fixtures.floats("golden/mel_filterbank.f32")
        let actual = MelSpectrogram.melFilterbank(
            sampleRate: 16_000, nFFT: 512, nMels: 80, fMin: 0, fMax: 8_000)

        XCTAssertEqual(actual.count, expected.count)
        let worst = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(worst, 1e-6, "slaney filterbank diverged from librosa")
    }

    func testMatchesReferenceFeatures() throws {
        let mel = MelSpectrogram()
        for name in ["sine440", "noise", "chirp", "short"] {
            let input = try Fixtures.floats("golden/\(name).input.f32")
            let expected = try Fixtures.floats("golden/\(name).expected.f32")
            let (actual, frames) = mel(input)

            XCTAssertEqual(actual.count, expected.count, "\(name): feature count")
            XCTAssertEqual(frames, expected.count / 80, "\(name): frame count")
            guard actual.count == expected.count else { continue }

            let worst = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
            XCTAssertLessThan(worst, 1e-3, "\(name) diverged by \(worst)")
        }
    }

    func testNormalizationIsZeroMeanPerBin() throws {
        let rate = 16_000
        let tone = (0..<rate).map { 0.5 * sin(2 * Float.pi * 440 * Float($0) / Float(rate)) }
        let (features, frames) = MelSpectrogram()(tone)
        XCTAssertEqual(frames, 101)

        for bin in 0..<80 {
            let row = features[(bin * frames)..<((bin + 1) * frames)]
            XCTAssertEqual(row.reduce(0, +) / Float(frames), 0, accuracy: 1e-3,
                           "mel bin \(bin) is not zero-mean")
        }
    }

    func testReflectPadMatchesNumpy() {
        // np.pad([1,2,3,4,5], 2, mode="reflect")
        XCTAssertEqual(MelSpectrogram.reflectPad([1, 2, 3, 4, 5], by: 2),
                       [3, 2, 1, 2, 3, 4, 5, 4, 3])
    }
}
