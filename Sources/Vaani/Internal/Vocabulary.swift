//
//  Vocabulary.swift
//  Vaani
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import Foundation

/// Per-language token table and the mask that narrows the global CTC logits.
///
/// `vocab.json[lang]` holds 257 tokens with blank at 256, and
/// `language_masks.json[lang]` is a 5633-long mask whose true positions are
/// `[i*256 ..< i*256+256] + [5632]` for language index `i`. Compacting by the
/// mask therefore lands exactly on the token table, and the RNNT head emits
/// those same 257 logits, so both decoders share one table.
struct Vocabulary {
    static let blank = 256
    static let globalSize = 5633

    /// RNNT start token. `config.json` says 256 and is wrong: AI4Bharat's own
    /// code never reads it and uses 5632. Measured on FLEURS Hindi, 5632 gives
    /// WER 0.083 against 0.125 for 256, which drops the leading token.
    static let startOfSequence = 5632

    let language: Language
    let tokens: [String]
    let maskIndices: [Int32]

    init(language: Language, assetsDirectory: URL) throws {
        self.language = language

        let tokensByLanguage = try JSONDecoder().decode(
            [String: [String]].self,
            from: Data(contentsOf: assetsDirectory.appendingPathComponent("vocab.json")))
        guard let tokens = tokensByLanguage[language.code] else {
            throw SpeechError.unsupportedLanguage(language.code)
        }
        guard tokens.count == 257 else {
            throw SpeechError.malformedModel(
                "vocab.json[\(language.code)] has \(tokens.count) tokens, expected 257")
        }

        let masksByLanguage = try JSONDecoder().decode(
            [String: [Bool]].self,
            from: Data(contentsOf: assetsDirectory.appendingPathComponent("language_masks.json")))
        guard let mask = masksByLanguage[language.code], mask.count == Self.globalSize else {
            throw SpeechError.malformedModel(
                "language_masks.json[\(language.code)] is not \(Self.globalSize) entries")
        }

        let indices = mask.enumerated().compactMap { $1 ? Int32($0) : nil }
        guard indices.count == 257 else {
            throw SpeechError.malformedModel(
                "mask for \(language.code) selects \(indices.count) entries, expected 257")
        }

        self.tokens = tokens
        self.maskIndices = indices
    }

    /// SentencePiece-style: U+2581 marks a word boundary.
    func text<S: Sequence>(from ids: S) -> String where S.Element == Int {
        var out = ""
        for id in ids where id != Self.blank && tokens.indices.contains(id) {
            out += tokens[id]
        }
        return out.replacingOccurrences(of: "\u{2581}", with: " ")
                  .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
