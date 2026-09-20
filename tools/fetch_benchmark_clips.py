#!/usr/bin/env python3
"""Fetch the 25 FLEURS Hindi clips the demo's benchmark runs against.

They are not committed: SwiftPM clones the whole repository for every consumer,
and 10 MB of benchmark audio is not something every dependent app should pay
for. Three clips live in Fixtures/ for the tests and the demo's sample list;
this fetches the full set into Fixtures/benchmark/.

    python tools/fetch_benchmark_clips.py

FLEURS is published by Google under CC BY 4.0.
"""
import argparse
import io
import json
import pathlib
import sys

URL = ("https://huggingface.co/api/datasets/google/fleurs/parquet"
       "/{config}/test/0.parquet")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Fixtures/benchmark")
    ap.add_argument("--config", default="hi_in", help="FLEURS config, e.g. ta_in")
    ap.add_argument("--count", type=int, default=25)
    args = ap.parse_args()

    try:
        import fsspec
        import pyarrow.parquet as pq
        import soundfile as sf
    except ImportError as e:
        print(f"missing dependency: {e}. pip install -r tools/requirements.txt",
              file=sys.stderr)
        return 1

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    url = URL.format(config=args.config)
    with fsspec.filesystem("http").open(url, "rb") as handle:
        table = pq.ParquetFile(handle).read_row_group(
            0, columns=["audio", "transcription"])

    manifest = []
    lang = args.config.split("_")[0]
    for i in range(min(args.count, table.num_rows)):
        audio = table["audio"][i].as_py()
        text = table["transcription"][i].as_py()
        samples, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(1)
        name = f"{lang}_{i}.wav"
        sf.write(out / name, samples, rate, subtype="PCM_16")
        manifest.append({"file": name, "lang": lang, "sr": rate,
                         "duration_s": round(len(samples) / rate, 2),
                         "reference": text})

    (out / "benchmark.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    total = sum(m["duration_s"] for m in manifest)
    print(f"wrote {len(manifest)} clips ({total:.0f}s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
