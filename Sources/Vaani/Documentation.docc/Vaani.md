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

The encoder emits one frame per 80 ms, and neither decoder is told which
frame produced which character. They resolve that differently.

``SpeechRecognizer/Decoder/ctc`` predicts a token at every frame plus a
blank, then collapses repeats and drops blanks. Each frame is predicted
independently of the others, so it is purely acoustic with no sense of which
token tends to follow which. Decoding is an argmax per frame, which is why it
is fast.

``SpeechRecognizer/Decoder/rnnt`` adds a prediction network — a 2-layer LSTM
that sees the tokens emitted so far, effectively a small language model — and
a joint network combining it with the encoder output. Decoding becomes a loop
that emits tokens until it emits blank, so the runtime is invoked once per
symbol rather than once per clip. That usually buys accuracy and always costs
speed: roughly 3 to 6 times slower, plus 42 MB of graphs.

CTC is the default. Reach for the transducer only once you have measured it
winning on your own audio.

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
