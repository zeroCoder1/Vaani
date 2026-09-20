#!/usr/bin/env python3
"""Dump the I/O contract of every ONNX graph in the model repo.

Reads graph structure only (external weight shards are not loaded), so this
works even with `fetch_model.py --small-only`.

    python tools/inspect_graphs.py --src models/src --json models/graphs.json
"""
import argparse
import json
import pathlib
import sys
from collections import Counter

import onnx
from onnx import TensorProto


def _dtype(t: int) -> str:
    return TensorProto.DataType.Name(t) if t else "?"


def _shape(vi) -> list:
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return ["<unranked>"]
    out = []
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            out.append(d.dim_value)
        elif d.HasField("dim_param"):
            out.append(d.dim_param)
        else:
            out.append("?")
    return out


def describe(path: pathlib.Path) -> dict:
    m = onnx.load(str(path), load_external_data=False)
    g = m.graph
    ops = Counter(n.op_type for n in g.node)
    return {
        "file": path.name,
        "ir_version": m.ir_version,
        "opset": [{"domain": o.domain or "ai.onnx", "version": o.version} for o in m.opset_import],
        "inputs": [
            {"name": i.name, "dtype": _dtype(i.type.tensor_type.elem_type), "shape": _shape(i)}
            for i in g.input
        ],
        "outputs": [
            {"name": o.name, "dtype": _dtype(o.type.tensor_type.elem_type), "shape": _shape(o)}
            for o in g.output
        ],
        "num_nodes": len(g.node),
        "num_initializers": len(g.initializer),
        "top_ops": ops.most_common(12),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/src")
    ap.add_argument("--json", default="models/graphs.json")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    if not src.exists():
        print(f"missing {src} - run tools/fetch_model.py first", file=sys.stderr)
        return 1

    files = sorted(src.rglob("*.onnx"))
    if not files:
        print(f"no .onnx under {src}", file=sys.stderr)
        return 1

    # collapse the 23 per-language heads to one representative
    seen_post_net = False
    report = []
    for f in files:
        if f.name.startswith("joint_post_net_"):
            if seen_post_net:
                continue
            seen_post_net = True
        try:
            report.append(describe(f))
        except Exception as e:  # noqa: BLE001
            print(f"  !! {f.name}: {type(e).__name__}: {e}", file=sys.stderr)

    for d in report:
        print(f"\n=== {d['file']} ===")
        print(f"  opset={d['opset']} nodes={d['num_nodes']} inits={d['num_initializers']}")
        print("  INPUTS:")
        for i in d["inputs"]:
            print(f"    {i['name']:<28} {i['dtype']:<8} {i['shape']}")
        print("  OUTPUTS:")
        for o in d["outputs"]:
            print(f"    {o['name']:<28} {o['dtype']:<8} {o['shape']}")
        print(f"  ops: {d['top_ops']}")

    out = pathlib.Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")

    # vocab / language mask summary
    for name in ("vocab.json", "language_masks.json"):
        for p in src.rglob(name):
            try:
                data = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            if name == "vocab.json":
                print(f"\n{name}: {len(data)} entries; sample {list(data.items())[:5] if isinstance(data, dict) else data[:5]}")
            else:
                keys = list(data) if isinstance(data, dict) else []
                print(f"\n{name}: {len(keys)} languages -> {keys}")
                if keys:
                    v = data[keys[0]]
                    n = len(v) if hasattr(v, "__len__") else "?"
                    print(f"  mask[{keys[0]}] len={n} type={type(v).__name__}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
