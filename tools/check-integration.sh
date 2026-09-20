#!/usr/bin/env bash
# Build a throwaway app that consumes Vaani the way a third party would.
#
# The demo app deliberately does NOT do this - it builds the library as its own
# framework target, because an app nested inside the package it depends on
# confuses Xcode. So this script is what proves SPM integration still works.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/Sources/Consumer"
cat > "$WORK/Package.swift" <<EOF
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Consumer",
    platforms: [.iOS(.v16), .macOS(.v14)],
    dependencies: [.package(path: "$REPO")],
    targets: [.executableTarget(name: "Consumer",
                                dependencies: [.product(name: "Vaani",
                                                        package: "$(basename "$REPO")")])]
)
EOF

cat > "$WORK/Sources/Consumer/main.swift" <<'EOF'
import Foundation
import Vaani

let models = URL(fileURLWithPath: CommandLine.arguments[1])
let audio = URL(fileURLWithPath: CommandLine.arguments[2])

let asr = try SpeechRecognizer(modelsAt: models, language: .hindi)
let result = try asr.transcribe(contentsOf: audio)
print("text: \(result.text)")
print("rtf:  \(String(format: "%.3f", result.realTimeFactor))")
EOF

MODELS="$REPO/models/int8_nc"
AUDIO="$REPO/Fixtures/hi_0.wav"

cd "$WORK"
if [ -f "$MODELS/encoder.onnx" ]; then
  swift run Consumer "$MODELS" "$AUDIO" 2>&1 | grep -vE '^\[|Compiling|Build complete|Fetch|Comput|Creating|Working|Download'
else
  echo "no quantized model, compiling only"
  swift build 2>&1 | tail -1
fi
echo "integration OK"
