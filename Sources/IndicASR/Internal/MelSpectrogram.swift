//
//  MelSpectrogram.swift
//  IndicASR
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import Accelerate
import Foundation

/// 80-bin log-mel features, matching NeMo's `AudioToMelSpectrogramPreprocessor`.
///
/// Two conventions here are load-bearing and fail silently if changed:
/// the window is a *symmetric* Hann (`torch.hann_window(periodic: false)`), not
/// the periodic one librosa defaults to, and the filterbank is slaney-scaled
/// (`htk: false`). Getting either wrong shifts features enough to cost WER
/// without ever raising an error.
struct MelSpectrogram {

    struct Config {
        var sampleRate = 16_000
        var winLength = 400          // 25 ms
        var hopLength = 160          // 10 ms
        var nFFT = 512
        var nMels = 80
        var fMin: Float = 0
        var fMax: Float = 8_000
        var preemphasis: Float = 0.97
        var logGuard: Float = 0x1p-24
        var normalizePerFeature = true
    }

    let config: Config
    private let window: [Float]
    private let filterbank: [Float]
    private let bins: Int
    private let log2n: vDSP_Length
    private let fft: FFTSetup

    init(config: Config = Config()) {
        precondition(config.nFFT > 0 && config.nFFT & (config.nFFT - 1) == 0)
        precondition(config.winLength <= config.nFFT)

        self.config = config
        bins = config.nFFT / 2 + 1
        log2n = vDSP_Length(log2(Double(config.nFFT)).rounded())

        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("vDSP_create_fftsetup failed for nFFT=\(config.nFFT)")
        }
        fft = setup

        var w = [Float](repeating: 0, count: config.nFFT)
        let offset = (config.nFFT - config.winLength) / 2
        for i in 0..<config.winLength {
            w[offset + i] = 0.5 - 0.5 * cos(2 * .pi * Float(i) / Float(config.winLength - 1))
        }
        window = w
        filterbank = Self.melFilterbank(sampleRate: config.sampleRate, nFFT: config.nFFT,
                                        nMels: config.nMels, fMin: config.fMin, fMax: config.fMax)
    }

    /// - Parameter samples: mono, [-1, 1], at `config.sampleRate`
    /// - Returns: row-major (nMels, frames)
    func callAsFunction(_ samples: [Float]) -> (features: [Float], frames: Int) {
        guard !samples.isEmpty else { return ([], 0) }

        var signal = samples
        if config.preemphasis != 0 {
            var previous = samples[0]
            for i in 1..<signal.count {
                let current = samples[i]
                signal[i] = current - config.preemphasis * previous
                previous = current
            }
        }

        let padded = Self.reflectPad(signal, by: config.nFFT / 2)
        let frames = 1 + (padded.count - config.nFFT) / config.hopLength
        guard frames > 0 else { return ([], 0) }

        var power = [Float](repeating: 0, count: bins * frames)
        var real = [Float](repeating: 0, count: config.nFFT / 2)
        var imaginary = [Float](repeating: 0, count: config.nFFT / 2)
        var windowed = [Float](repeating: 0, count: config.nFFT)

        for t in 0..<frames {
            let start = t * config.hopLength
            for i in 0..<config.nFFT { windowed[i] = padded[start + i] * window[i] }

            real.withUnsafeMutableBufferPointer { re in
                imaginary.withUnsafeMutableBufferPointer { im in
                    var split = DSPSplitComplex(realp: re.baseAddress!, imagp: im.baseAddress!)
                    windowed.withUnsafeBufferPointer { src in
                        src.baseAddress!.withMemoryRebound(to: DSPComplex.self,
                                                           capacity: config.nFFT / 2) {
                            vDSP_ctoz($0, 2, &split, 1, vDSP_Length(config.nFFT / 2))
                        }
                    }
                    vDSP_fft_zrip(fft, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                }
            }

            // vDSP packs DC in real[0] and Nyquist in imaginary[0], scaled by 2.
            let dc = real[0] * 0.5, nyquist = imaginary[0] * 0.5
            power[t] = dc * dc
            power[(bins - 1) * frames + t] = nyquist * nyquist
            for k in 1..<(config.nFFT / 2) {
                let re = real[k] * 0.5, im = imaginary[k] * 0.5
                power[k * frames + t] = re * re + im * im
            }
        }

        var mel = [Float](repeating: 0, count: config.nMels * frames)
        vDSP_mmul(filterbank, 1, power, 1, &mel, 1,
                  vDSP_Length(config.nMels), vDSP_Length(frames), vDSP_Length(bins))

        var guardValue = config.logGuard
        vDSP_vsadd(mel, 1, &guardValue, &mel, 1, vDSP_Length(mel.count))
        var count = Int32(mel.count)
        vvlogf(&mel, mel, &count)

        if config.normalizePerFeature && frames > 1 {
            normalize(&mel, rows: config.nMels, columns: frames)
        }
        return (mel, frames)
    }

    /// Zero mean, unit variance per mel bin, with NeMo's ddof = 1.
    private func normalize(_ mel: inout [Float], rows: Int, columns: Int) {
        mel.withUnsafeMutableBufferPointer { buffer in
            for m in 0..<rows {
                let row = buffer.baseAddress! + m * columns
                var mean: Float = 0
                vDSP_meanv(row, 1, &mean, vDSP_Length(columns))
                var negated = -mean
                vDSP_vsadd(row, 1, &negated, row, 1, vDSP_Length(columns))
                var sumOfSquares: Float = 0
                vDSP_svesq(row, 1, &sumOfSquares, vDSP_Length(columns))
                var scale = 1 / (sqrt(sumOfSquares / Float(columns - 1)) + 1e-5)
                vDSP_vsmul(row, 1, &scale, row, 1, vDSP_Length(columns))
            }
        }
    }

    static func reflectPad(_ x: [Float], by pad: Int) -> [Float] {
        guard pad > 0 else { return x }
        precondition(x.count > pad, "signal shorter than the reflection width")
        var out = [Float]()
        out.reserveCapacity(x.count + 2 * pad)
        for i in stride(from: pad, to: 0, by: -1) { out.append(x[i]) }
        out.append(contentsOf: x)
        for i in 1...pad { out.append(x[x.count - 1 - i]) }
        return out
    }
}

// Slaney mel scale, equivalent to librosa.filters.mel(htk: false, norm: "slaney").
extension MelSpectrogram {
    private static let hzPerMel = 200.0 / 3
    private static let logThresholdHz = 1000.0
    private static let logThresholdMel = 1000.0 / (200.0 / 3)
    private static let logStep = log(6.4) / 27

    static func hzToMel(_ hz: Double) -> Double {
        hz >= logThresholdHz
            ? logThresholdMel + log(hz / logThresholdHz) / logStep
            : hz / hzPerMel
    }

    static func melToHz(_ mel: Double) -> Double {
        mel >= logThresholdMel
            ? logThresholdHz * exp(logStep * (mel - logThresholdMel))
            : mel * hzPerMel
    }

    static func melFilterbank(sampleRate: Int, nFFT: Int, nMels: Int,
                              fMin: Float, fMax: Float) -> [Float] {
        let bins = nFFT / 2 + 1
        let fftFreqs = (0..<bins).map {
            Double($0) * Double(sampleRate) / 2 / Double(bins - 1)
        }
        let low = hzToMel(Double(fMin)), high = hzToMel(Double(fMax))
        let edges = (0..<(nMels + 2)).map {
            melToHz(low + (high - low) * Double($0) / Double(nMels + 1))
        }

        var fb = [Float](repeating: 0, count: nMels * bins)
        for m in 0..<nMels {
            let (lower, center, upper) = (edges[m], edges[m + 1], edges[m + 2])
            let area = 2 / (upper - lower)   // slaney: equal area per filter
            for b in 0..<bins {
                let f = fftFreqs[b]
                let rise = (f - lower) / (center - lower)
                let fall = (upper - f) / (upper - center)
                fb[m * bins + b] = Float(max(0, min(rise, fall)) * area)
            }
        }
        return fb
    }
}
