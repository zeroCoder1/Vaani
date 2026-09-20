# ``Vaani``

Offline speech-to-text for 22 Indian languages, on device.

## Overview

Vaani runs AI4Bharat's IndicConformer 600M through ONNX Runtime. Audio
never leaves the device and no network is needed once the model is cached.

```swift
import Vaani

let asr = try SpeechRecognizer(modelsAt: directory, language: .hindi)
let result = try asr.transcribe(contentsOf: audioURL)
print(result.text, result.realTimeFactor)
```

Loading pulls roughly 900 MB of weights into memory and
``SpeechRecognizer/transcribe(_:using:)`` is blocking, so keep both off the
main thread.

### Getting the model

The weights are too large to ship inside an app bundle, so
``ModelDownloader`` fetches them on first use and caches them afterwards.

```swift
let asr = try await SpeechRecognizer.downloading(
    from: URL(string: "https://cdn.example.com/vaani")!,
    language: .hindi
) { progress in
    print("\(progress.file) \(Int(progress.fraction * 100))%")
}
```

Downloads are offline-first: once the files are cached nothing touches the
network, so a later launch works on a plane. Only missing files are fetched,
and because the encoder is shared across every language, adding a second
language costs about 0.17 MB rather than another 900 MB.

### Choosing a decoder

``SpeechRecognizer/Decoder/ctc`` is the default and is what you want unless
you have a reason otherwise. ``SpeechRecognizer/Decoder/rnnt`` can be
slightly more accurate on some audio but runs a search step per symbol, so
it is several times slower.

Loading both shares the one encoder rather than paying for it twice:

```swift
let asr = try SpeechRecognizer(modelsAt: directory,
                               language: .hindi,
                               decoders: [.ctc, .rnnt])
let quick = try asr.transcribe(samples, using: .ctc)
let careful = try asr.transcribe(samples, using: .rnnt)
```

### Recording

``MicrophoneCapture`` records straight into the format the encoder expects.

```swift
let mic = MicrophoneCapture()
guard await mic.requestPermission() else { return }
try await mic.start()
// ...
let result = try asr.transcribe(await mic.stop())
```

Add `NSMicrophoneUsageDescription` to your Info.plist, or the app is
terminated on first use.

## Topics

### Recognising speech

- ``SpeechRecognizer``
- ``Language``

### Loading the model

- ``ModelDownloader``

### Audio input

- ``AudioFile``
- ``MicrophoneCapture``

### Errors

- ``SpeechError``
