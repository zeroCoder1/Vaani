# Vaani

[![Swift 5.9](https://img.shields.io/badge/Swift-5.9-orange.svg)](https://swift.org)
[![Platforms](https://img.shields.io/badge/platforms-iOS%2016%20%7C%20macOS%2014-lightgrey.svg)](https://developer.apple.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Offline speech-to-text for 22 Indian languages on iOS, running AI4Bharat's
[IndicConformer 600M](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
through ONNX Runtime. No network at inference time.

For how the model was shrunk from 2.4 GB to something a phone will download,
and what each script in `tools/` does, see
[docs/MODEL_PIPELINE.md](docs/MODEL_PIPELINE.md).

## Install

Add the package in Xcode via **File → Add Package Dependencies**, or in a
`Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/zeroCoder1/Vaani", from: "1.0.0")
],
targets: [
    .target(name: "YourApp", dependencies: [
        .product(name: "Vaani", package: "Vaani")
    ])
]
```

Note the two names: the product is `Vaani`, but `package:` is the
repository name, `Vaani`. Swift Package Manager derives the package
identifier from the URL, not from the product.

Requires iOS 16 / macOS 14. The only dependency is ONNX Runtime.

## Use

```swift
import Vaani

let asr = try SpeechRecognizer(modelsAt: modelDirectory, language: .hindi)
let result = try asr.transcribe(contentsOf: audioURL)
print(result.text, result.realTimeFactor)
```

Loading pulls ~900 MB of weights into memory and `transcribe` blocks, so keep
both off the main thread.

To fetch the model on first launch instead of shipping it:

```swift
let asr = try await SpeechRecognizer.downloading(
    from: URL(string: "https://cdn.example.com/vaani")!,
    language: .hindi
) { progress in
    print("\(progress.file) \(Int(progress.fraction * 100))%")
}
```

`ModelDownloader` is offline-first — once the files are cached it never touches
the network, so a second launch works on a plane. It also only downloads what is
missing, which matters because the 870 MB encoder is shared by every language
and a per-language transducer head is 0.17 MB. Adding Tamil to an app that
already has Hindi costs 0.17 MB.

From the microphone:

```swift
let mic = MicrophoneCapture()
guard await mic.requestPermission() else { return }
try mic.start()
// ...
let result = try asr.transcribe(mic.stop())
```

That needs `NSMicrophoneUsageDescription` in your Info.plist.

## Decoders

Both decoders turn a sequence of audio frames into text. The encoder emits one
frame per 80 ms, so a 9-second clip is 115 frames while the transcript might be
40 characters — neither decoder is told which frame produced which character,
and they resolve that differently.

`.ctc` is the default. Use `.rnnt` only if you have measured it winning on
your audio.

```swift
let asr = try SpeechRecognizer(modelsAt: directory,
                               language: .hindi,
                               decoders: [.ctc, .rnnt])
let quick = try asr.transcribe(samples, using: .ctc)
let careful = try asr.transcribe(samples, using: .rnnt)
```

Loading both shares the one encoder rather than paying for it twice.

### CTC

Connectionist Temporal Classification predicts a token at *every* frame, plus
a special blank, then collapses the result:

```
frames:           h  h  _  e  _  l  l  _  l  o     (_ = blank)
collapse repeats: h     _  e  _  l     _  l  o
drop blanks:      h        e     l        l  o     -> "hello"
```

The blank is what makes double letters possible. Without one between the two
`l`s, collapsing repeats would merge them.

The property that matters: **each frame is predicted independently** given the
audio. There is no mechanism for "having just emitted q, u is likely next", so
it is purely acoustic. Decoding is an argmax per frame and some array work,
which is why it is fast.

### RNNT

The transducer adds two pieces so that output tokens can depend on each other:

- a **prediction network** — here a 2-layer LSTM, hidden size 640 — which sees
  the tokens emitted so far, effectively a small built-in language model
- a **joint network** combining what the audio says at frame *t* with what has
  been written so far

Decoding becomes a loop: at each frame, emit tokens until the model emits
blank, then advance. `maxSymbolsPerFrame` caps that at 10 so a model that
never emits blank cannot spin forever.

That loop is why the search lives in Swift rather than in the graph. ONNX
cannot express it, so the runtime is invoked once per emitted symbol instead
of once per clip.

### Which to use

| | CTC | RNNT |
| --- | --- | --- |
| real-time factor (Mac) | ~0.04 | ~0.11–0.27 |
| extra download | — | 42 MB, plus 0.17 MB per language |
| WER on `hi_0.wav` | 0.042 | 0.083 |

Transducers usually win on accuracy because of that built-in language
modelling. On the single clip measured here CTC came out ahead, but one
9-second sample is not evidence — run both across a real set before choosing.
CTC is the default mainly for the 3–6x speed difference, which is far more
noticeable on a phone than on a desktop.

## Shipping this in your own app

The demo defaults to `http://localhost:8000` and carries two local-network
Info.plist keys. Both exist only so you can test the download flow against
`tools/dev.sh serve` without paying for hosting. **Neither belongs in a
production app.**

For a real app:

```swift
let asr = try await SpeechRecognizer.downloading(
    from: URL(string: "https://cdn.yourcompany.com/vaani")!,
    language: .hindi
)
```

and do *not* copy these from the demo's `project.yml`:

| key | why the demo has it | your app |
| --- | --- | --- |
| `NSAppTransportSecurity.NSAllowsLocalNetworking` | ATS blocks plain HTTP to a LAN IP | drop it — HTTPS satisfies ATS |
| `NSLocalNetworkUsageDescription` | iOS prompts before LAN access | drop it — a CDN is not the local network |

Shipping `NSAllowsLocalNetworking` loosens ATS for no benefit once you are on
HTTPS. `NSMicrophoneUsageDescription` you do still need, if you use
`MicrophoneCapture`.

What you host is the contents of `models/int8_nc/` — the `.onnx` files, their
`.data` sidecars, `vocab.json`, `language_masks.json`, and the `manifest.json`
from `tools/make_manifest.py`. Any static host works; no server logic is
involved, `ModelDownloader` just issues GETs. Run `make_manifest.py` *without*
`--no-hash` before shipping so downloads are checksum-verified.

## Preparing the model

The upstream weights are fp32 and gated. Accept the terms on the model page,
then:

```bash
python3 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt
export HF_TOKEN=hf_...
.venv/bin/python tools/fetch_model.py
.venv/bin/python tools/quantize.py --src models/src --out models/int8_nc \
    --mode int8 --ops MatMul Gemm
.venv/bin/python tools/make_manifest.py --src models/int8_nc
```

Host `models/int8_nc/` on any static server and point `ModelDownloader` at it.
AI4Bharat's repo is gated, so an app cannot download from it directly.

`tools/dev.sh stage` assembles exactly the files a host needs into
`models/upload/`, with checksums. See
[docs/MODEL_PIPELINE.md](docs/MODEL_PIPELINE.md) for what each step does.

### What quantization costs

25 FLEURS Hindi clips, 318 s of audio, greedy CTC:

| build | size | WER | RTF |
| --- | --- | --- | --- |
| fp32 | 2556 MB | 0.1034 | 0.082 |
| int8, MatMul + Gemm + Conv | 707 MB | 0.1380 | 0.041 |
| int8, MatMul + Gemm | 952 MB | 0.1056 | 0.032 |

Quantizing Conv is what costs accuracy, not int8. Leaving Conv alone is nearly
lossless and happens to be the fastest of the three, so that is the default.
The smaller build is there if you need the 245 MB back.

## Demo

`Examples/VaaniDemo` exercises the whole surface: all 22 languages, both
decoders, bundled clips with ground-truth transcripts, an audio file picker, and
the microphone. It reports real-time factor and WER per run.

```bash
tools/dev.sh xcode         # reset Xcode state and open the demo project
tools/dev.sh run           # build, install, launch on the simulator
tools/dev.sh push-model    # side-load the model, no server needed
```

Open `Examples/VaaniDemo/VaaniDemo.xcodeproj`.

The demo builds `Sources/Vaani` as its own framework target rather than
depending on the repo root as a local Swift package. An app nested inside the
package it depends on makes Xcode enumerate the entire repository — `.venv` and
`models/` included, about 5 GB — and it breaks outright if the repo folder is
also open, with "Missing package product 'Vaani'". Building the sources
directly sidesteps all of that.

That means the demo does not itself exercise SPM, so there is a separate check
for the path a third party actually takes:

```bash
tools/check-integration.sh
```

It builds a throwaway package that depends on this one by path and transcribes a
clip through it.

Simulator builds need no code signing. For a device, either pick your team in
Xcode's Signing tab, or set it before generating the project:

```bash
export DEVELOPMENT_TEAM=YOURTEAMID
tools/dev.sh xcode
```

`project.yml` leaves `DEVELOPMENT_TEAM` empty on purpose so a fresh clone is not
tied to one account.

On a physical device the model has to arrive over the network:

```bash
tools/dev.sh serve         # prints the LAN URL to paste into the app
```

Two Info.plist keys make that work and fail silently without it:
`NSAppTransportSecurity.NSAllowsLocalNetworking`, because ATS exempts
`localhost` but not a LAN IP, and `NSLocalNetworkUsageDescription`, because
iOS 14+ denies local network access with no usage string. The demo also sets
`com.apple.developer.kernel.increased-memory-limit`; without it 900 MB of
weights gets the app jetsammed on an 8 GB device.

## Three things that will bite you

All three fail silently — nothing throws, the output just gets worse. Full
detail and the measurements behind them are in
[docs/MODEL_PIPELINE.md](docs/MODEL_PIPELINE.md#three-findings-that-cost-real-measurement).

- The analysis window is a **symmetric** Hann, not the periodic one librosa
  and scipy default to. The wrong one shifts features by ~0.17.
- `config.json` says `SOS: 256` and is **wrong**; the transducer start token
  is 5632. Following the file drops the leading token.
- Transducer feedback uses **local** per-language token ids, even though the
  embedding is over the global vocabulary. Remapping to global ids collapses
  decoding into garbage.

## Tools

| script | what it does |
| --- | --- |
| `tools/fetch_model.py` | download the gated repo; `--small-only` skips the weight shards |
| `tools/inspect_graphs.py` | dump every graph's I/O names, shapes and opset |
| `tools/quantize.py` | fp16 or int8, with `--ops` to choose what gets quantized |
| `tools/reference_frontend.py` | numpy log-mel reference and golden vectors |
| `tools/run_reference.py` | numpy/ORT decoder used as the parity reference and WER harness |
| `tools/make_manifest.py` | generate `manifest.json` for `ModelDownloader` |
| `tools/stage_upload.py` | assemble `models/upload/` with exactly what a host needs |
| `tools/dev.sh` | build, test, run, serve, push-model, stage, xcode |
| `tools/check-integration.sh` | build a throwaway consumer against the package over SPM |

## Documentation

The package ships a DocC catalog, so symbols carry documentation in Xcode's
Quick Help and the documentation viewer. To build the archive:

```bash
tools/dev.sh docs
```

DocC is built for iOS. ONNX Runtime's Objective-C headers include C++ that
clang cannot parse when extracting macOS symbol graphs, so a macOS docs build
fails inside the dependency; `.spi.yml` pins Swift Package Index to iOS for
the same reason.

## Tests

```bash
tools/dev.sh test
```

Ten tests. The frontend ones run anywhere; the recognizer ones skip unless a
quantized model is present, and check Swift output against `run_reference.py` on
the same weights. They assert on drift and WER rather than exact strings,
because vDSP's FFT and numpy's differ by ~3.5e-05 — enough to flip a near-tied
argmax about once every few hundred frames.

## A note on this checkout

The repo sits under `~/Documents`, which is iCloud-synced. iCloud stamps
`com.apple.FinderInfo` on files and `codesign` refuses to sign a bundle carrying
it, so SwiftPM's in-tree `.build` fails when signing test bundles. `tools/dev.sh`
builds to a scratch directory outside the synced tree. Xcode is unaffected.
