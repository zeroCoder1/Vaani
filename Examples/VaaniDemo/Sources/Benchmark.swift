//
//  Benchmark.swift
//  VaaniDemo
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import Foundation
import Vaani

/// One decoder's results across the whole clip set.
struct BenchmarkRun: Identifiable {
    let id = UUID()
    let decoder: SpeechRecognizer.Decoder
    let clips: Int
    let audioSeconds: Double
    let processingSeconds: Double
    let meanWER: Double
    let medianWER: Double

    var realTimeFactor: Double {
        audioSeconds > 0 ? processingSeconds / audioSeconds : 0
    }
}

enum Benchmark {
    /// The full clip set, separate from the three shown in the main list.
    ///
    /// Empty when the clips have not been fetched. Deliberately does not fall
    /// back to the bundled samples: benchmarking three clips and presenting
    /// the result as twenty-five is worse than reporting nothing.
    static let clips: [Sample] = {
        guard let url = Bundle.main.url(forResource: "benchmark", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let list = try? JSONDecoder().decode([Sample].self, from: data)
        else { return [] }
        return list
    }()

    static func summarise(decoder: SpeechRecognizer.Decoder,
                          results: [(audio: Double, processing: Double, wer: Double)]) -> BenchmarkRun {
        let wers = results.map(\.wer).sorted()
        let median: Double
        if wers.isEmpty {
            median = 0
        } else if wers.count % 2 == 1 {
            median = wers[wers.count / 2]
        } else {
            median = (wers[wers.count / 2 - 1] + wers[wers.count / 2]) / 2
        }
        return BenchmarkRun(
            decoder: decoder,
            clips: results.count,
            audioSeconds: results.reduce(0) { $0 + $1.audio },
            processingSeconds: results.reduce(0) { $0 + $1.processing },
            meanWER: results.isEmpty ? 0 : results.reduce(0) { $0 + $1.wer } / Double(results.count),
            medianWER: median)
    }
}
