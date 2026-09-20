# Fixtures

Shared test data. Read from disk by the test suite and bundled into the demo
app, so there is one copy rather than one per consumer.

## Audio

`hi_*.wav` and `samples.json` are three Hindi clips from
[FLEURS](https://huggingface.co/datasets/google/fleurs) with their reference
transcripts, extracted by `tools/` from the `hi_in` test split.

FLEURS is published by Google under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). These clips are
redistributed here under that licence; the rest of this repository is under its
own licence.

## Generated

`golden/*.f32` are log-mel reference vectors from
`tools/reference_frontend.py --dump-golden Fixtures/golden`, which is
cross-validated against librosa and AI4Bharat's TorchScript preprocessor.

`python_reference.json` holds `tools/run_reference.py` output for the clips
above, used to check the Swift decoder against the numpy one.

Both regenerate from the tools; they are committed so the tests run without a
model download.
