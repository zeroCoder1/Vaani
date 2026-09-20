//
//  Fixtures.swift
//  VaaniTests
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import Foundation
import XCTest

/// Test data lives in `Fixtures/` at the repo root rather than being bundled,
/// so the demo app and the tests share one copy. These tests only ever run from
/// a checkout, which is the same assumption `modelDirectory` already makes.
enum Fixtures {

    static let root: URL? = {
        var directory = URL(fileURLWithPath: #filePath)
        for _ in 0..<6 {
            directory = directory.deletingLastPathComponent()
            let candidate = directory.appendingPathComponent("Fixtures")
            if FileManager.default.fileExists(
                atPath: candidate.appendingPathComponent("samples.json").path) {
                return candidate
            }
        }
        return nil
    }()

    static func url(_ relativePath: String) throws -> URL {
        guard let root else { throw XCTSkip("Fixtures/ not found; run from a checkout") }
        let url = root.appendingPathComponent(relativePath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("missing fixture \(relativePath)")
        }
        return url
    }

    static func floats(_ relativePath: String) throws -> [Float] {
        try Data(contentsOf: try url(relativePath)).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
    }

    static func decode<T: Decodable>(_ type: T.Type, from relativePath: String) throws -> T {
        try JSONDecoder().decode(type, from: Data(contentsOf: try url(relativePath)))
    }

    /// Where the quantized model lives. `VAANI_MODEL_DIR` overrides it.
    static func modelDirectory() throws -> URL {
        let manager = FileManager.default
        func holdsModel(_ url: URL) -> Bool {
            manager.fileExists(atPath: url.appendingPathComponent("encoder.onnx").path)
        }
        if let override = ProcessInfo.processInfo.environment["VAANI_MODEL_DIR"] {
            let url = URL(fileURLWithPath: override)
            if holdsModel(url) { return url }
        }
        var directory = URL(fileURLWithPath: #filePath)
        for _ in 0..<6 {
            directory = directory.deletingLastPathComponent()
            let candidate = directory.appendingPathComponent("models/int8_nc")
            if holdsModel(candidate) { return candidate }
        }
        throw XCTSkip("no quantized model; run tools/quantize.py first")
    }
}
