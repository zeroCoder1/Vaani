# IndicASR

Offline speech-to-text for 22 Indian languages on iOS, running AI4Bharat's
[IndicConformer 600M](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
through ONNX Runtime. No network at inference time.

## Install

```swift
.package(url: "https://github.com/<you>/IndicASR", from: "1.0.0")
```

Requires iOS 16 / macOS 14.

## Use

```swift
import IndicASR

let asr = try SpeechRecognizer(modelsAt: modelDirectory, language: .hindi)
let result = try asr.transcribe(contentsOf: audioURL)
print(result.text, result.realTimeFactor)
```

Loading pulls ~900 MB of weights into memory and `transcribe` blocks, so keep
both off the main thread.

To fetch the model on first launch instead of shipping it:

```swift
let asr = try await SpeechRecognizer.downloading(
    from: URL(string: "https://cdn.example.com/indicasr")!,
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

## Shipping this in your own app

The demo defaults to `http://localhost:8000` and carries two local-network
Info.plist keys. Both exist only so you can test the download flow against
`tools/dev.sh serve` without paying for hosting. **Neither belongs in a
production app.**

For a real app:

```swift
let asr = try await SpeechRecognizer.downloading(
    from: URL(string: "https://cdn.yourcompany.com/indicasr")!,
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

`Examples/IndicASRDemo` exercises the whole surface: all 22 languages, both
decoders, bundled clips with ground-truth transcripts, an audio file picker, and
the microphone. It reports real-time factor and WER per run.

```bash
tools/dev.sh xcode         # reset Xcode state and open the demo project
tools/dev.sh run           # build, install, launch on the simulator
tools/dev.sh push-model    # side-load the model, no server needed
```

Open `Examples/IndicASRDemo/IndicASRDemo.xcodeproj`.

The demo builds `Sources/IndicASR` as its own framework target rather than
depending on the repo root as a local Swift package. An app nested inside the
package it depends on makes Xcode enumerate the entire repository — `.venv` and
`models/` included, about 5 GB — and it breaks outright if the repo folder is
also open, with "Missing package product 'IndicASR'". Building the sources
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

All three fail silently — nothing throws, the output just gets worse.

The analysis window is a *symmetric* Hann, `torch.hann_window(periodic: false)`,
not the periodic Hann librosa and scipy default to. Using the wrong one shifts
normalized features by about 0.17. With the right one the Swift frontend matches
AI4Bharat's TorchScript preprocessor to 3.5e-05 across the realistic amplitude
range.

`config.json` gives `SOS: 256` and it is wrong. The transducer start token is
5632. AI4Bharat's own code never reads `config.json` — `from_pretrained` builds
the config from kwargs and takes the 5632 default — so their path works by
accident. Following the file drops the leading token: WER 0.125 against 0.083.

Transducer token feedback uses local, per-language ids (0…255), even though
`prediction.embed.weight` is `(5633, 640)`. Remapping to global ids
(`languageIndex * 256 + token`) looks more principled and collapses decoding into
repeated garbage, WER above 3.

## Tools

| script | what it does |
| --- | --- |
| `tools/fetch_model.py` | download the gated repo; `--small-only` skips the weight shards |
| `tools/inspect_graphs.py` | dump every graph's I/O names, shapes and opset |
| `tools/quantize.py` | fp16 or int8, with `--ops` to choose what gets quantized |
| `tools/reference_frontend.py` | numpy log-mel reference and golden vectors |
| `tools/run_reference.py` | numpy/ORT decoder used as the parity reference and WER harness |
| `tools/make_manifest.py` | generate `manifest.json` for `ModelDownloader` |
| `tools/dev.sh` | build, test, run, serve, push-model, xcode |
| `tools/check-integration.sh` | build a throwaway consumer against the package over SPM |

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
