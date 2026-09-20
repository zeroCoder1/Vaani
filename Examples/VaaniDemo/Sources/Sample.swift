//
//  Sample.swift
//  VaaniDemo
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import Vaani
import Foundation

/// A bundled FLEURS clip with its ground-truth transcript, so the demo can
/// report a real WER rather than just "some text came out".
struct Sample: Codable, Identifiable, Hashable {
    let file: String
    let lang: String
    let durationS: Double
    let reference: String

    var id: String { file }
    var url: URL? { Bundle.main.url(forResource: file, withExtension: nil) }
    var language: Language? { Language(rawValue: lang) }

    enum CodingKeys: String, CodingKey {
        case file, lang, reference
        case durationS = "duration_s"
    }

    static let bundled: [Sample] = {
        guard let url = Bundle.main.url(forResource: "samples", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let samples = try? JSONDecoder().decode([Sample].self, from: data)
        else { return [] }
        return samples
    }()
}

func errorRate(reference: String, hypothesis: String) -> Double {
    let a = reference.split(separator: " ").map(String.init)
    let b = hypothesis.split(separator: " ").map(String.init)
    guard !a.isEmpty else { return b.isEmpty ? 0 : 1 }

    var previous = Array(0...b.count)
    var current = [Int](repeating: 0, count: b.count + 1)
    for i in 1...a.count {
        current[0] = i
        for j in 1...max(b.count, 1) where !b.isEmpty {
            current[j] = min(previous[j] + 1, current[j - 1] + 1,
                             previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1))
        }
        swap(&previous, &current)
    }
    return Double(previous[b.count]) / Double(a.count)
}
