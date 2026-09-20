#!/usr/bin/env python3
"""Assemble exactly the files a host needs into models/upload/.

Reads manifest.json to decide what is required, so it stays correct if the
profiles change. Uses APFS clones where available, so staging ~950 MB is
instant and costs no extra disk.

    python tools/make_manifest.py --src models/int8_nc   # checksums first
    python tools/stage_upload.py --src models/int8_nc
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/int8_nc")
    ap.add_argument("--out", default="models/upload")
    args = ap.parse_args()

    src, out = pathlib.Path(args.src), pathlib.Path(args.out)
    manifest = src / "manifest.json"
    if not manifest.exists():
        print(f"{manifest} missing - run tools/make_manifest.py first", file=sys.stderr)
        return 1

    m = json.loads(manifest.read_text())
    if any(e.get("sha256") is None for e in m["shared"].values()):
        print("warning: manifest has no checksums; rerun make_manifest.py "
              "without --no-hash before shipping", file=sys.stderr)

    needed = set(m["profiles"]["ctc"]) | set(m["profiles"]["rnnt"]) | {"manifest.json"}
    # perLanguage is keyed by language code, each holding that language's files.
    for files in m["perLanguage"].values():
        needed |= set(files)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    total = 0
    for name in sorted(needed):
        source = src / name
        if not source.exists():
            print(f"  missing {name}", file=sys.stderr)
            continue
        subprocess.run(["cp", "-c", str(source), str(out / name)], check=False)
        if not (out / name).exists():
            shutil.copy2(source, out / name)
        total += source.stat().st_size

    extra = sorted({p.name for p in src.iterdir() if p.is_file()} - needed)
    print(f"staged {len(needed)} files, {total/1e6:.0f} MB -> {out}")
    if extra:
        print(f"not needed for hosting: {', '.join(extra)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
