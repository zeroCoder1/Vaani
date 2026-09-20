//
//  ONNX.swift
//  IndicASR
//
//  Created by Shrutesh Sharma on 20/09/2026.
//

import Foundation
import OnnxRuntimeBindings

protocol ONNXScalar {}
extension Float: ONNXScalar {}
extension Int32: ONNXScalar {}
extension Int64: ONNXScalar {}

/// ORT returns untyped `NSMutableData`, so shape handling lives here rather
/// than being repeated (and mis-cast) at every call site.
enum ONNX {

    static func session(env: ORTEnv, path: URL, threads: Int32) throws -> ORTSession {
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw SpeechError.missingModelFile(path.lastPathComponent)
        }
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(threads)
        try options.setGraphOptimizationLevel(.all)
        return try ORTSession(env: env, modelPath: path.path, sessionOptions: options)
    }

    static func tensor(_ values: [Float], shape: [Int]) throws -> ORTValue {
        guard values.count == shape.reduce(1, *) else {
            throw SpeechError.shapeMismatch(
                "expected \(shape.reduce(1, *)) floats for shape \(shape), got \(values.count)")
        }
        return try value(values, .float, shape)
    }

    static func tensor(_ values: [Int32], shape: [Int]) throws -> ORTValue {
        try value(values, .int32, shape)
    }

    static func tensor(_ values: [Int64], shape: [Int]) throws -> ORTValue {
        try value(values, .int64, shape)
    }

    /// Restricted to the trivial scalar types ORT tensors actually hold, so the
    /// raw copy below is sound.
    private static func value<T: ONNXScalar>(_ values: [T],
                                             _ type: ORTTensorElementDataType,
                                             _ shape: [Int]) throws -> ORTValue {
        let data = values.withUnsafeBytes {
            NSMutableData(bytes: $0.baseAddress, length: $0.count)
        }
        return try ORTValue(tensorData: data, elementType: type,
                            shape: shape.map(NSNumber.init(value:)))
    }

    static func shape(of value: ORTValue) throws -> [Int] {
        try value.tensorTypeAndShapeInfo().shape.map(\.intValue)
    }

    static func floats(_ value: ORTValue) throws -> [Float] {
        let data = try value.tensorData() as Data
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    static func int64s(_ value: ORTValue) throws -> [Int64] {
        let data = try value.tensorData() as Data
        return data.withUnsafeBytes { Array($0.bindMemory(to: Int64.self)) }
    }

    static func run(_ session: ORTSession, inputs: [String: ORTValue],
                    outputs: [String]) throws -> [ORTValue] {
        let produced = try session.run(withInputs: inputs,
                                       outputNames: Set(outputs),
                                       runOptions: nil)
        return try outputs.map {
            guard let value = produced[$0] else {
                throw SpeechError.malformedModel("no output named '\($0)'")
            }
            return value
        }
    }
}
