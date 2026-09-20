#!/usr/bin/env python3
"""Shrink the ONNX graphs for on-device use.

  fp16  ~2x smaller, near-lossless, good default for 8GB iPhones
  int8  ~4x smaller (encoder ~2.4GB -> ~600MB), dynamic weight quantization
        of MatMul/Gemm only; activations stay float. Validate WER after.

    python tools/quantize.py --src models/src --out models/int8 --mode int8
    python tools/quantize.py --src models/src --out models/fp16 --mode fp16

Note: the encoder holds ~600M params as external data. Peak host RAM during
conversion is roughly 3x the fp32 size (~8GB) - run it on the Mac, not in CI.
"""
import argparse
import pathlib
import shutil
import sys
import time

import onnx


def convert_fp16(src: pathlib.Path, dst: pathlib.Path, keep_io_fp32: bool) -> None:
    from onnxconverter_common import float16

    m = onnx.load(str(src))  # pulls in external data
    m16 = float16.convert_float_to_float16(
        m,
        keep_io_types=keep_io_fp32,
        disable_shape_infer=True,
    )
    onnx.save(
        m16,
        str(dst),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=dst.name + ".data",
        convert_attribute=True,
    )


def convert_int8(src: pathlib.Path, dst: pathlib.Path, per_channel: bool,
                 ops: list[str]) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=False,
        # MatMul/Gemm carry ~2.1GB of the encoder's 2.4GB; adding Conv picks up
        # the remaining ~0.3GB of pointwise convs. LayerNorm/Softmax always stay
        # float - quantizing them wrecks ASR accuracy.
        op_types_to_quantize=ops,
        extra_options={"MatMulConstBOnly": True},
        use_external_data_format=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["fp16", "int8"], required=True)
    ap.add_argument("--per-channel", action="store_true", default=True)
    ap.add_argument("--keep-io-fp32", action="store_true", default=True,
                    help="fp16 mode: keep graph inputs/outputs float32")
    ap.add_argument("--ops", nargs="*", default=["MatMul", "Gemm"],
                    help="op types to quantize, e.g. --ops MatMul Gemm Conv")
    ap.add_argument("--only", nargs="*", default=None,
                    help="only convert these filenames, e.g. encoder.onnx")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    out = pathlib.Path(args.out)
    if not src.exists():
        print(f"missing {src} - run tools/fetch_model.py first", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    models = sorted(src.rglob("*.onnx"))
    if args.only:
        models = [m for m in models if m.name in set(args.only)]
    if not models:
        print("nothing to convert", file=sys.stderr)
        return 1

    total_in = total_out = 0
    for p in models:
        dst = out / p.name
        t0 = time.time()
        # size of the graph plus any external shards sitting beside it
        before = p.stat().st_size
        print(f"[{args.mode}] {p.name} ...", flush=True)
        try:
            if args.mode == "fp16":
                convert_fp16(p, dst, args.keep_io_fp32)
            else:
                convert_int8(p, dst, args.per_channel, args.ops)
        except Exception as e:  # noqa: BLE001
            print(f"  !! {type(e).__name__}: {e}", file=sys.stderr)
            continue
        after = sum(f.stat().st_size for f in out.glob(dst.name + "*"))
        total_in += before
        total_out += after
        print(f"  {before/1e6:.1f}MB -> {after/1e6:.1f}MB  ({time.time()-t0:.1f}s)")

    # non-onnx assets ride along unchanged
    for name in ("vocab.json", "language_masks.json", "config.json"):
        for p in src.rglob(name):
            shutil.copy2(p, out / name)
            break

    print(f"\ntotal graph bytes {total_in/1e6:.1f}MB -> {total_out/1e6:.1f}MB")
    print(f"output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
