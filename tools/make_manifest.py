#!/usr/bin/env python3
"""Emit the manifest.json that ModelStore downloads against.

Upload the quantized directory plus the generated manifest.json to any static
host (S3, R2, a public HF repo) and point `ModelStore(baseURL:)` at it.

    python tools/make_manifest.py --src models/int8_nc --variant int8

sha256 over ~870MB takes a few seconds; pass --no-hash to skip during dev.
"""
import argparse
import hashlib
import json
import pathlib
import sys

# Files each decoding profile needs, before .data companions are added.
CTC_CORE = ["encoder.onnx", "ctc_decoder.onnx"]
RNNT_CORE = ["encoder.onnx", "rnnt_decoder.onnx", "joint_enc.onnx",
             "joint_pred.onnx", "joint_pre_net.onnx"]
JSON_ASSETS = ["vocab.json", "language_masks.json"]
PER_LANGUAGE = ["joint_post_net_{lang}.onnx"]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def expand(src: pathlib.Path, names: list[str]) -> list[str]:
    """Add the `.data` sidecar for any graph that stores weights externally."""
    out = []
    for n in names:
        if not (src / n).exists():
            print(f"  warn: {n} not found, skipping", file=sys.stderr)
            continue
        out.append(n)
        if (src / (n + ".data")).exists():
            out.append(n + ".data")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/int8_nc")
    ap.add_argument("--variant", default="int8")
    ap.add_argument("--version", default="1.0")
    ap.add_argument("--no-hash", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    if not src.is_dir():
        print(f"missing {src}", file=sys.stderr)
        return 1

    ctc = expand(src, CTC_CORE) + JSON_ASSETS
    rnnt = expand(src, RNNT_CORE) + JSON_ASSETS
    shared_names = sorted(set(ctc) | set(rnnt))

    shared = {}
    for n in shared_names:
        p = src / n
        if not p.exists():
            print(f"  warn: {n} missing", file=sys.stderr)
            continue
        shared[n] = {"size": p.stat().st_size,
                     "sha256": None if args.no_hash else sha256(p)}

    per_lang = {}
    for pattern in PER_LANGUAGE:
        sample = pattern.replace("{lang}", "hi")
        for name in expand(src, [sample]):
            key = name.replace("_hi.onnx", "_{lang}.onnx")
            p = src / name
            per_lang[key] = {"size": p.stat().st_size,
                             "sha256": None if args.no_hash else sha256(p)}

    manifest = {
        "version": args.version,
        "variant": args.variant,
        "shared": shared,
        "perLanguage": per_lang,
        "profiles": {"ctc": [n for n in ctc if n in shared],
                     "rnnt": [n for n in rnnt if n in shared]},
    }

    out = pathlib.Path(args.out) if args.out else src / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))

    def total(names):
        return sum(shared[n]["size"] for n in names if n in shared)

    print(f"wrote {out}")
    print(f"  ctc  profile: {total(manifest['profiles']['ctc'])/1e6:8.1f} MB")
    print(f"  rnnt profile: {total(manifest['profiles']['rnnt'])/1e6:8.1f} MB"
          f"  + {sum(e['size'] for e in per_lang.values())/1e6:.2f} MB per language")
    if args.no_hash:
        print("  (checksums skipped - rerun without --no-hash before shipping)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
