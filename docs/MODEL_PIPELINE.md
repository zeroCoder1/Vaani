# Model pipeline

How AI4Bharat's IndicConformer 600M went from a 2.4 GB research release to
something an iPhone downloads and runs offline, and what each script in
`tools/` is for.

## Why this needed work

The upstream release is already ONNX, which removes the usual
NeMo-to-ONNX export step. What it is not is shippable:

- **2.4 GB of fp32 weights.** No phone is downloading that, and it will not
  fit in an app's memory budget.
- **Decomposed into separate graphs.** Encoder, CTC decoder, transducer
  decoder, joint network and 22 per-language output heads are separate
  files wired together by the caller. There is no single model to load.
- **The reference implementation is Python.** It depends on `torch`,
  `transformers` and a TorchScript audio frontend, none of which exist on
  iOS.
- **Gated.** The weights require accepting terms on Hugging Face, so an app
  cannot fetch them directly.

So the work splits in two: shrink the model without wrecking accuracy, and
reimplement everything around it in Swift.

## The model

| component | role |
| --- | --- |
| `encoder.onnx` | 24-layer Conformer, d_model 1024, ~600M parameters |
| `ctc_decoder.onnx` | encoder output to 5633 global logits |
| `rnnt_decoder.onnx` | 2-layer LSTM prediction network, hidden 640 |
| `joint_enc` / `joint_pred` / `joint_pre_net` | transducer joint network |
| `joint_post_net_<lang>.onnx` × 22 | 257 per-language logits |
| `vocab.json`, `language_masks.json` | token tables and masks |

The encoder subsamples by 8, so one output frame covers 80 ms.

The per-language head design is the useful part: one 880 MB encoder is
shared by every language, and switching language swaps a 0.17 MB head. It is
also why the model does not drop into an off-the-shelf runtime, since
generic importers assume a single output head.

### Vocabulary layout

Determined by inspecting the assets, not assumed:

- `vocab.json[lang]` is exactly 257 tokens, with blank at index 256, spelled
  `"|"` in every language.
- `language_masks.json[lang]` is a 5633-long boolean mask whose true
  positions are exactly `[i*256 ..< i*256+256] + [5632]`, where `i` is the
  language's index.
- Compacting the global CTC logits by that mask therefore lands exactly on
  the language's token table, and the transducer head already emits those
  same 257 logits.

Both decoders share one token table and one blank index.

## The pipeline

### 1. Fetch — `fetch_model.py`

Downloads the gated repository. `--small-only` skips the 2.4 GB of weight
shards and pulls just the graph structure and JSON, which is enough to map
the interfaces before committing to the full download.

### 2. Inspect — `inspect_graphs.py`

Dumps every graph's input and output names, shapes and opset without loading
external data. This is what produced the table above, and what revealed that
`prediction.embed.weight` is `(5633, 640)` while the joint head emits 257 —
the detail behind the transducer token-feedback finding below.

### 3. Quantize — `quantize.py`

Produces fp16 or int8 builds. `--ops` selects which operators are quantised,
which turned out to matter more than the bit width.

Measured on 25 FLEURS Hindi clips, 318 seconds of audio, greedy CTC:

| build | size | WER | RTF (Mac) |
| --- | --- | --- | --- |
| fp32 | 2556 MB | 0.1034 | 0.082 |
| int8, MatMul + Gemm + Conv | 707 MB | 0.1380 | 0.041 |
| **int8, MatMul + Gemm** | **952 MB** | **0.1056** | **0.032** |

**Quantising Conv is what costs accuracy, not int8.** Including Conv saves
another 245 MB but costs 3.5 points of WER — a third worse, relatively.
Leaving Conv in float is very nearly lossless and is also the fastest of the
three, so that is the default. LayerNorm and Softmax are never quantised.

### 4. Build the reference — `reference_frontend.py`, `run_reference.py`

Before porting anything to Swift there has to be something to check it
against.

`reference_frontend.py` reimplements NeMo's `FilterbankFeatures` in numpy and
is cross-validated against librosa — the filterbank matches to 1.9e-09. It
also emits golden vectors that the Swift tests assert against, so the Swift
frontend is checked against numpy, which is checked against librosa, which
is checked against the TorchScript preprocessor AI4Bharat ships.

`run_reference.py` is a pure numpy and ONNX Runtime decoder implementing both
CTC and transducer decoding with no torch or transformers. It serves as the
golden reference for the Swift port and as the WER harness for the
quantisation table above.

### 5. Package for hosting — `make_manifest.py`, `stage_upload.py`

`make_manifest.py` writes the `manifest.json` that `ModelDownloader` reads:
per-file sizes, SHA-256 checksums, and which files each decoding profile
needs. `--no-hash` skips hashing during development; ship with checksums so a
truncated download fails loudly instead of producing a model that loads and
emits nonsense.

`stage_upload.py` assembles `models/upload/` containing exactly the files a
host needs and nothing else, using APFS clones so staging 950 MB is instant.

| profile | size |
| --- | --- |
| CTC | 906 MB |
| transducer | 925 MB + 0.17 MB per language |

## Three findings that cost real measurement

All three fail silently. Nothing throws; the output just gets worse.

### The analysis window

NeMo uses a **symmetric** Hann window, `torch.hann_window(periodic: false)`.
librosa and scipy default to the **periodic** one. Using the wrong window
shifts normalised features by about 0.17 — enough to cost WER with no error
anywhere. With the correct window the Swift frontend matches AI4Bharat's
TorchScript preprocessor to **3.5e-05** across the amplitude range speech
occupies.

### `config.json` gives the wrong SOS

`config.json` declares `SOS: 256`. It is wrong; the transducer start token is
**5632**. AI4Bharat's own code never reads `config.json` — `from_pretrained`
builds its config from keyword arguments and falls back to the 5632 default —
so the reference works by accident.

| start token | WER |
| --- | --- |
| 5632 | 0.083 |
| 256 | 0.125 (drops the leading token) |

### Transducer feedback uses local token ids

`prediction.embed.weight` is `(5633, 640)` — the global vocabulary — while
the joint head emits 257 per-language logits. The apparently principled move
is to map the predicted token back to global space
(`languageIndex * 256 + token`) before feeding it to the prediction network.

That collapses decoding into repeated garbage at WER above 3. Feeding the
local id straight back is correct.

## Reproducing it

```bash
python3 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt
export HF_TOKEN=hf_...                       # after accepting the gate

.venv/bin/python tools/fetch_model.py
.venv/bin/python tools/quantize.py --src models/src --out models/int8_nc \
    --mode int8 --ops MatMul Gemm
tools/dev.sh stage                           # manifest + models/upload/
```

Then upload `models/upload/` to any static host and point
`ModelDownloader(source:)` at it.

To re-derive the numbers above:

```bash
.venv/bin/python tools/run_reference.py --model models/int8_nc --lang hi \
    --decoding ctc --limit 25
```

## Why the tools are in Python at all

They run once, on a Mac, to produce artefacts. ONNX Runtime's quantisation
tooling, the reference NeMo frontend and the Hugging Face client are all
Python, and reimplementing any of them in Swift would add work while making
the reference less trustworthy — the point of `run_reference.py` is that it
is an *independent* implementation to check the Swift one against.

Nothing in `tools/` ships in the app. The iOS target depends only on ONNX
Runtime.
