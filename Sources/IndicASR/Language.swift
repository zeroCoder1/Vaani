//
//  Language.swift
//  IndicASR
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import Foundation

/// The 22 scheduled languages the model was trained on.
///
/// Declaration order is significant: a case's index is its block offset in the
/// 5633-token global vocabulary, and it matches the key order in `vocab.json`.
public enum Language: String, CaseIterable, Sendable, Codable, Identifiable {
    case assamese = "as", bengali = "bn", bodo = "brx", dogri = "doi"
    case konkani = "kok", gujarati = "gu", hindi = "hi", kannada = "kn"
    case kashmiri = "ks", maithili = "mai", malayalam = "ml", marathi = "mr"
    case manipuri = "mni", nepali = "ne", odia = "or", punjabi = "pa"
    case sanskrit = "sa", santali = "sat", sindhi = "sd", tamil = "ta"
    case telugu = "te", urdu = "ur"

    /// Stable identity for SwiftUI lists and pickers.
    public var id: String { rawValue }

    /// ISO 639 code, as used in the model's filenames and asset keys.
    public var code: String { rawValue }

    public var name: String {
        switch self {
        case .assamese: "Assamese";   case .bengali: "Bengali"
        case .bodo: "Bodo";           case .dogri: "Dogri"
        case .konkani: "Konkani";     case .gujarati: "Gujarati"
        case .hindi: "Hindi";         case .kannada: "Kannada"
        case .kashmiri: "Kashmiri";   case .maithili: "Maithili"
        case .malayalam: "Malayalam"; case .marathi: "Marathi"
        case .manipuri: "Manipuri";   case .nepali: "Nepali"
        case .odia: "Odia";           case .punjabi: "Punjabi"
        case .sanskrit: "Sanskrit";   case .santali: "Santali"
        case .sindhi: "Sindhi";       case .tamil: "Tamil"
        case .telugu: "Telugu";       case .urdu: "Urdu"
        }
    }

    var vocabularyOffset: Int { Self.allCases.firstIndex(of: self)! * 256 }
}
