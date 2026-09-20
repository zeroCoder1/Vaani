//
//  AudioFile.swift
//  IndicASR
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import AVFoundation
import Foundation

/// Decodes audio files into the mono 16 kHz samples the encoder expects.
public enum AudioFile {

    public static let sampleRate: Double = 16_000

    /// Read any format AVFoundation supports, downmixed to mono and resampled.
    public static func samples(at url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let frames = AVAudioFrameCount(file.length)
        guard frames > 0 else { throw SpeechError.emptyAudio }

        guard let input = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else {
            throw SpeechError.unsupportedAudioFormat("could not allocate a \(frames)-frame buffer")
        }
        try file.read(into: input)

        if format.sampleRate == sampleRate, format.channelCount == 1,
           let channel = input.floatChannelData {
            return Array(UnsafeBufferPointer(start: channel[0], count: Int(input.frameLength)))
        }
        return try resample(input, from: format)
    }

    private static func resample(_ input: AVAudioPCMBuffer,
                                 from format: AVAudioFormat) throws -> [Float] {
        guard let target = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                         sampleRate: sampleRate,
                                         channels: 1,
                                         interleaved: false),
              let converter = AVAudioConverter(from: format, to: target) else {
            throw SpeechError.unsupportedAudioFormat(
                "cannot convert \(Int(format.sampleRate))Hz/\(format.channelCount)ch to 16kHz mono")
        }
        converter.sampleRateConverterQuality = AVAudioQuality.high.rawValue

        let capacity = AVAudioFrameCount(
            Double(input.frameLength) * sampleRate / format.sampleRate) + 4096
        guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else {
            throw SpeechError.unsupportedAudioFormat("could not allocate the output buffer")
        }

        var consumed = false
        var failure: NSError?
        let status = converter.convert(to: output, error: &failure) { _, outStatus in
            if consumed {
                outStatus.pointee = .endOfStream
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return input
        }
        if let failure {
            throw SpeechError.unsupportedAudioFormat(failure.localizedDescription)
        }
        guard status != .error, let channel = output.floatChannelData else {
            throw SpeechError.unsupportedAudioFormat("resampling failed")
        }
        return Array(UnsafeBufferPointer(start: channel[0], count: Int(output.frameLength)))
    }
}
