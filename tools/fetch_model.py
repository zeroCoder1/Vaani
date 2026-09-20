#!/usr/bin/env python3
"""Download the gated ai4bharat/indic-conformer-600m-multilingual repo.

Requires a Hugging Face token with access granted on the model page:
    https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual

    export HF_TOKEN=hf_xxx        # or: huggingface-cli login
    python tools/fetch_model.py
"""
import argparse
import os
import sys

REPO = "ai4bharat/indic-conformer-600m-multilingual"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/src", help="destination directory")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument(
        "--small-only",
        action="store_true",
        help="skip the ~2.4GB encoder weights; fetch graphs + json + decoders only",
    )
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or True  # True -> fall back to cached login

    ignore = None
    if args.small_only:
        # external weight shards are the bulk; graph protos and json are tiny
        ignore = ["assets/onnx__MatMul_*", "assets/layers.*", "assets/onnx__Conv_*",
                  "assets/Constant_*", "assets/pre_encode.*"]

    try:
        path = snapshot_download(
            repo_id=args.repo,
            local_dir=args.out,
            token=token,
            ignore_patterns=ignore,
        )
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        print(f"FAILED: {type(e).__name__}: {msg[:400]}", file=sys.stderr)
        if "401" in msg or "restricted" in msg or "gated" in msg.lower():
            print(
                "\nThis repo is gated. Accept the terms while logged in at:\n"
                f"  https://huggingface.co/{args.repo}\n"
                "then set HF_TOKEN (Settings -> Access Tokens, 'read' scope).",
                file=sys.stderr,
            )
        return 1

    print(f"downloaded -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
