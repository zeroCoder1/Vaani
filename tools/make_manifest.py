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


def verify(manifest: dict, src: pathlib.Path) -> None:
    """Check the manifest against the files it describes.

    An earlier version templated the per-language entries and filled them from
    Hindi, so every other language carried Hindi's checksum. Sizes are identical
    across languages, so nothing caught it until a download failed on a device.
    """
    problems = []

    def check(name: str, entry: dict) -> None:
        path = src / name
        if not path.exists():
            problems.append(f"{name}: missing")
            return
        if path.stat().st_size != entry["size"]:
            problems.append(f"{name}: size {path.stat().st_size} != {entry['size']}")
        if entry.get("sha256") and sha256(path) != entry["sha256"]:
            problems.append(f"{name}: checksum does not match the file")

    for name, entry in manifest["shared"].items():
        check(name, entry)

    seen: dict[str, str] = {}
    for lang, files in manifest["perLanguage"].items():
        for name, entry in files.items():
            check(name, entry)
            digest = entry.get("sha256")
            if digest and digest in seen and seen[digest] != name:
                problems.append(
                    f"{name} shares a checksum with {seen[digest]}; "
                    "per-language files must each carry their own")
            if digest:
                seen[digest] = name

    if problems:
        raise SystemExit("manifest is wrong:\n  " + "\n  ".join(problems))
    print(f"  verified {len(manifest['shared'])} shared and "
          f"{sum(len(f) for f in manifest['perLanguage'].values())} per-language files")


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

    # Keyed by language, then by real filename. An earlier version stored one
    # templated entry per pattern and filled it from Hindi, which gave every
    # other language Hindi's checksum: sizes are identical across languages, so
    # only the hash caught it, and only once hashing was switched on.
    per_lang: dict[str, dict] = {}
    for pattern in PER_LANGUAGE:
        for path in sorted(src.glob(pattern.replace("{lang}", "*"))):
            stem = path.name.split(".onnx")[0]
            lang = stem.rsplit("_", 1)[-1]
            entry = {"size": path.stat().st_size,
                     "sha256": None if args.no_hash else sha256(path)}
            per_lang.setdefault(lang, {})[path.name] = entry
            data = path.with_name(path.name + ".data")
            if data.exists():
                per_lang[lang][data.name] = {
                    "size": data.stat().st_size,
                    "sha256": None if args.no_hash else sha256(data)}

    manifest = {
        "version": args.version,
        "variant": args.variant,
        "shared": shared,
        "perLanguage": per_lang,
        "profiles": {"ctc": [n for n in ctc if n in shared],
                     "rnnt": [n for n in rnnt if n in shared]},
    }

    verify(manifest, src)

    out = pathlib.Path(args.out) if args.out else src / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))

    def total(names):
        return sum(shared[n]["size"] for n in names if n in shared)

    print(f"wrote {out}")
    print(f"  ctc  profile: {total(manifest['profiles']['ctc'])/1e6:8.1f} MB")
    per_language_bytes = max(
        (sum(e["size"] for e in files.values()) for files in per_lang.values()),
        default=0)
    print(f"  rnnt profile: {total(manifest['profiles']['rnnt'])/1e6:8.1f} MB"
          f"  + {per_language_bytes/1e6:.2f} MB per language"
          f"  ({len(per_lang)} languages)")
    if args.no_hash:
        print("  (checksums skipped - rerun without --no-hash before shipping)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
