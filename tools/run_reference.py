#!/usr/bin/env python3
"""Pure numpy + onnxruntime reference decoder for IndicConformer.

No torch, no transformers - this is the golden reference the Swift port is
validated against, and it doubles as the fp32-vs-int8 comparison harness.

    python tools/run_reference.py --model models/src --lang hi --decoding ctc
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from reference_frontend import LogMelFrontend  # noqa: E402

LANGS = ['as', 'bn', 'brx', 'doi', 'kok', 'gu', 'hi', 'kn', 'ks', 'mai', 'ml',
         'mr', 'mni', 'ne', 'or', 'pa', 'sa', 'sat', 'sd', 'ta', 'te', 'ur']
BLANK_LOCAL = 256       # blank within a language's 257-wide head
BLANK_GLOBAL = 5632     # shared blank in the 5633-wide global vocab
# RNNT SOS. config.json claims 256, which is WRONG: AI4Bharat's own code path
# (from_pretrained -> IndicASRConfig(**kwargs)) never reads config.json and uses
# the 5632 default. Measured on FLEURS hi: SOS=5632 -> WER 0.083,
# SOS=256 -> 0.125 (drops the leading token). Verified by A/B, not assumed.
SOS = 5632
# Token feedback into the prediction net uses LOCAL (per-language, 0..255) ids,
# even though prediction.embed.weight is (5633, 640). Remapping to global ids
# (lang_index*256 + k) collapses decoding to repeated garbage (WER 3.25).
MAX_SYMBOLS = 10
PRED_LAYERS, PRED_HIDDEN = 2, 640


def sess(path, threads=0):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


class IndicASR:
    def __init__(self, root: pathlib.Path, lang: str, need_rnnt: bool = True):
        self.root, self.lang = root, lang
        # fp32 repo nests under assets/; quantized output dirs are flat
        a = root / "assets" if (root / "assets").is_dir() else root
        self.fe = LogMelFrontend()
        self.encoder = sess(a / "encoder.onnx")
        self.ctc = sess(a / "ctc_decoder.onnx")
        self.vocab = json.loads((a / "vocab.json").read_text())[lang]
        masks = json.loads((a / "language_masks.json").read_text())[lang]
        self.mask_idx = np.array([i for i, v in enumerate(masks) if v], dtype=np.int64)
        self.lang_offset = LANGS.index(lang) * 256
        if need_rnnt:
            self.dec = sess(a / "rnnt_decoder.onnx")
            self.j_enc = sess(a / "joint_enc.onnx")
            self.j_pred = sess(a / "joint_pred.onnx")
            self.j_pre = sess(a / "joint_pre_net.onnx")
            self.j_post = sess(a / f"joint_post_net_{lang}.onnx")

    # ---- encoder -------------------------------------------------------
    def encode(self, wav: np.ndarray):
        feats = self.fe(wav)[None]                      # (1, 80, T)
        length = np.array([feats.shape[2]], dtype=np.int64)
        out, enc_len = self.encoder.run(
            ["outputs", "encoded_lengths"],
            {"audio_signal": feats.astype(np.float32), "length": length},
        )
        return out, enc_len                             # (1,1024,T'), (1,)

    # ---- CTC -----------------------------------------------------------
    def decode_ctc(self, enc_out):
        logprobs = self.ctc.run(["logprobs"], {"encoder_output": enc_out})[0]
        lp = logprobs[:, :, self.mask_idx]               # (1, T', 257)
        ids = lp[0].argmax(-1)
        # collapse repeats, then drop blanks
        keep = np.concatenate(([True], ids[1:] != ids[:-1]))
        ids = ids[keep]
        toks = [self.vocab[i] for i in ids if i != BLANK_LOCAL]
        return "".join(toks).replace("▁", " ").strip()

    # ---- RNNT ----------------------------------------------------------
    def decode_rnnt(self, enc_out, sos: int = SOS, global_feedback: bool = False):
        """Defaults are the measured-correct combination; args retained for A/B."""
        j_enc = self.j_enc.run(["output"], {"input": enc_out.transpose(0, 2, 1)})[0]
        hyp = []
        last = sos
        h = np.zeros((PRED_LAYERS, 1, PRED_HIDDEN), dtype=np.float32)
        c = np.zeros((PRED_LAYERS, 1, PRED_HIDDEN), dtype=np.float32)
        for t in range(j_enc.shape[1]):
            f = j_enc[:, t:t + 1, :]
            added = 0
            while added < MAX_SYMBOLS:
                g, _, nh, nc = self.dec.run(
                    ["outputs", "prednet_lengths", "states", "162"],
                    {"targets": np.array([[last]], dtype=np.int32),
                     "target_length": np.array([1], dtype=np.int32),
                     "states.1": h, "onnx::Slice_3": c},
                )
                gp = self.j_pred.run(["output"], {"input": g.transpose(0, 2, 1)})[0]
                jo = self.j_pre.run(["output"], {"input": f + gp})[0]
                logits = self.j_post.run(["output"], {"input": jo})[0]
                k = int(logits.reshape(-1).argmax())      # 0..256, per-language
                if k == BLANK_LOCAL:
                    break
                hyp.append(k)
                last = (self.lang_offset + k) if global_feedback else k
                h, c = nh, nc
                added += 1
        return "".join(self.vocab[i] for i in hyp).replace("▁", " ").strip()


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1,
                          d[i-1, j-1] + (r[i-1] != h[j-1]))
    return d[len(r), len(h)] / max(1, len(r))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/src")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--samples", default="samples/samples.json")
    ap.add_argument("--decoding", default="ctc", choices=["ctc", "rnnt", "both"])
    ap.add_argument("--rnnt-ab", action="store_true",
                    help="A/B the RNNT index-space hypotheses")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--out", default=None, help="write results json here")
    args = ap.parse_args()

    import soundfile as sf
    meta = json.loads(pathlib.Path(args.samples).read_text())[:args.limit]
    root = pathlib.Path(args.model)
    sdir = pathlib.Path(args.samples).parent

    t0 = time.time()
    m = IndicASR(root, args.lang, need_rnnt=args.decoding != "ctc")
    print(f"loaded in {time.time()-t0:.1f}s\n")

    results = []
    for item in meta:
        wav, sr = sf.read(sdir / item["file"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        t0 = time.time()
        enc, enc_len = m.encode(wav)
        t_enc = time.time() - t0
        dur = len(wav) / sr
        print(f"--- {item['file']}  {dur:.1f}s  enc {t_enc:.2f}s "
              f"(RTF {t_enc/dur:.3f})  frames {enc.shape[2]} "
              f"(subsample {int(round(len(wav)/160/enc.shape[2]))}x)")
        print(f"  REF : {item['reference'][:100]}")
        rec = {"file": item["file"], "reference": item["reference"],
               "enc_seconds": round(t_enc, 3), "audio_seconds": round(dur, 2)}

        if args.decoding in ("ctc", "both"):
            t0 = time.time()
            hyp = m.decode_ctc(enc)
            rec["ctc"] = hyp; rec["ctc_wer"] = round(wer(item["reference"], hyp), 4)
            print(f"  CTC : {hyp[:100]}")
            print(f"        WER {rec['ctc_wer']:.3f}   ({time.time()-t0:.2f}s)")

        if args.decoding in ("rnnt", "both"):
            variants = ([("global_fb sos=5632", 5632, True),
                         ("local_fb  sos=5632", 5632, False),
                         ("local_fb  sos=256", 256, False)]
                        if args.rnnt_ab else [("global_fb sos=5632", 5632, True)])
            for label, sos, gfb in variants:
                t0 = time.time()
                hyp = m.decode_rnnt(enc, sos, gfb)
                w = wer(item["reference"], hyp)
                rec[f"rnnt[{label}]"] = hyp
                rec[f"rnnt_wer[{label}]"] = round(w, 4)
                print(f"  RNNT[{label}]: {hyp[:90]}")
                print(f"        WER {w:.3f}   ({time.time()-t0:.2f}s)")
        results.append(rec)
        print()

    for k in ("ctc_wer",):
        vals = [r[k] for r in results if k in r]
        if vals:
            print(f"mean {k}: {np.mean(vals):.4f}")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
