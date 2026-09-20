# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-09-20

First release.

### Added

- `SpeechRecognizer` for offline transcription in 22 Indian languages, with
  greedy CTC and transducer decoding over AI4Bharat's IndicConformer 600M.
- `ModelDownloader` for fetching and caching model files on demand, so apps
  ship a small binary. Offline-first: once cached, nothing touches the
  network. Only missing files are downloaded, and because the encoder is
  shared across languages, adding a language costs about 0.17 MB.
- `MicrophoneCapture` for recording straight into the sample format the
  encoder expects.
- `AudioFile` for decoding and resampling any format AVFoundation supports.
- A log-mel frontend matching NeMo's `AudioToMelSpectrogramPreprocessor`,
  validated against AI4Bharat's TorchScript preprocessor to 3.5e-05.
- Python tooling under `tools/` for fetching, quantizing, evaluating and
  packaging the model, plus a numpy reference decoder used to verify the
  Swift implementation.
- A demo app under `Examples/` exercising every language, both decoders,
  bundled clips, file import and the microphone.
- DocC documentation catalog.

### Notes

The default build quantizes MatMul and Gemm to int8 but leaves Conv in
float. Measured on 25 FLEURS Hindi clips, that is 952 MB at 0.1056 WER
against 0.1034 for fp32 at 2556 MB. Including Conv saves a further 245 MB
but costs 3.5 points of WER.

[Unreleased]: https://github.com/zeroCoder1/Indic-languages/compare/1.0.0...HEAD
[1.0.0]: https://github.com/zeroCoder1/Indic-languages/releases/tag/1.0.0
