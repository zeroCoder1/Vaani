//
//  SpeechError.swift
//  Vaani
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import Foundation

/// Errors thrown by ``SpeechRecognizer``, ``ModelDownloader`` and ``AudioFile``.
public enum SpeechError: LocalizedError {
    case missingModelFile(String)
    case unsupportedLanguage(String)
    case malformedModel(String)
    case shapeMismatch(String)
    case emptyAudio
    case unsupportedAudioFormat(String)
    case downloadFailed(String)
    case checksumMismatch(file: String)

    /// A message suitable for showing to a person.
    public var errorDescription: String? {
        switch self {
        case .missingModelFile(let name):
            "Model file not found: \(name)"
        case .unsupportedLanguage(let code):
            "Language '\(code)' is not in this model"
        case .malformedModel(let detail):
            "Malformed model: \(detail)"
        case .shapeMismatch(let detail):
            "Tensor shape mismatch: \(detail)"
        case .emptyAudio:
            "Audio buffer was empty"
        case .unsupportedAudioFormat(let detail):
            "Unsupported audio: \(detail)"
        case .downloadFailed(let detail):
            "Model download failed: \(detail)"
        case .checksumMismatch(let file):
            "Checksum mismatch for \(file); the download was corrupted"
        }
    }
}
