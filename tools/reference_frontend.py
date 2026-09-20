#!/usr/bin/env python3
"""NeMo-compatible log-mel frontend in numpy + golden vectors for the Swift port.

This mirrors NeMo's AudioToMelSpectrogramPreprocessor / FilterbankFeatures.
Defaults below are NeMo's; `--from-ts assets/preprocessor.ts` overrides them
with the values actually baked into the shipped TorchScript preprocessor.

    python tools/reference_frontend.py --inspect-ts models/src/assets/preprocessor.ts
    python tools/reference_frontend.py --dump-golden Fixtures/golden
"""
import argparse
import json
import pathlib
import re
import sys
import zipfile

import numpy as np

# --- NeMo FilterbankFeatures defaults -------------------------------------
DEFAULTS = dict(
    sample_rate=16000,
    win_length=400,      # 25ms
    hop_length=160,      # 10ms
    n_fft=512,           # 2**ceil(log2(400))
    n_mels=80,
    fmin=0.0,
    fmax=8000.0,         # sr/2
    preemph=0.97,
    mag_power=2.0,
    log_zero_guard=2.0 ** -24,
    normalize="per_feature",
)


# --- slaney mel scale (librosa-equivalent, no librosa dependency) ---------
_F_SP = 200.0 / 3
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = np.log(6.4) / 27.0


def hz_to_mel(f):
    scalar = np.isscalar(f) or np.ndim(f) == 0
    f = np.atleast_1d(np.asarray(f, dtype=np.float64))
    mel = f / _F_SP
    hi = f >= _MIN_LOG_HZ
    mel[hi] = _MIN_LOG_MEL + np.log(f[hi] / _MIN_LOG_HZ) / _LOGSTEP
    return float(mel[0]) if scalar else mel


def mel_to_hz(m):
    scalar = np.isscalar(m) or np.ndim(m) == 0
    m = np.atleast_1d(np.asarray(m, dtype=np.float64))
    f = m * _F_SP
    hi = m >= _MIN_LOG_MEL
    f[hi] = _MIN_LOG_HZ * np.exp(_LOGSTEP * (m[hi] - _MIN_LOG_MEL))
    return float(f[0]) if scalar else f


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    """librosa.filters.mel(htk=False, norm='slaney') -> (n_mels, n_fft//2+1)"""
    n_bins = n_fft // 2 + 1
    fftfreqs = np.linspace(0, sr / 2.0, n_bins)
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    freqs = mel_to_hz(mel_pts)

    fdiff = np.diff(freqs)
    ramps = freqs[:, None] - fftfreqs[None, :]
    W = np.zeros((n_mels, n_bins))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        W[i] = np.maximum(0, np.minimum(lower, upper))
    # slaney normalization: equal area per filter
    enorm = 2.0 / (freqs[2:n_mels + 2] - freqs[:n_mels])
    return (W * enorm[:, None]).astype(np.float32)


def hann_periodic_false(n):
    """torch.hann_window(n, periodic=False) — NeMo uses periodic=False."""
    return np.hanning(n).astype(np.float32)


class LogMelFrontend:
    def __init__(self, **kw):
        self.p = {**DEFAULTS, **kw}
        p = self.p
        self.window = hann_periodic_false(p["win_length"])
        self.fb = mel_filterbank(
            p["sample_rate"], p["n_fft"], p["n_mels"], p["fmin"], p["fmax"]
        )

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        """float32 mono [-1,1] @16k -> (n_mels, T) float32"""
        p = self.p
        x = np.asarray(wav, dtype=np.float32)

        # preemphasis: y[0]=x[0]; y[t]=x[t]-a*x[t-1]
        if p["preemph"]:
            x = np.concatenate([x[:1], x[1:] - p["preemph"] * x[:-1]])

        n_fft, hop, win = p["n_fft"], p["hop_length"], p["win_length"]
        # center=True, pad_mode='reflect'
        x = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")

        n_frames = 1 + (len(x) - n_fft) // hop
        pad_l = (n_fft - win) // 2
        frames = np.lib.stride_tricks.as_strided(
            x, shape=(n_frames, n_fft), strides=(x.strides[0] * hop, x.strides[0])
        ).copy()
        w = np.zeros(n_fft, dtype=np.float32)
        w[pad_l:pad_l + win] = self.window
        frames *= w

        spec = np.fft.rfft(frames, n=n_fft, axis=1)
        power = (np.abs(spec) ** p["mag_power"]).astype(np.float32)   # (T, n_bins)
        mel = self.fb @ power.T                                      # (n_mels, T)
        mel = np.log(mel + p["log_zero_guard"])

        if p["normalize"] == "per_feature":
            m = mel.mean(axis=1, keepdims=True)
            s = mel.std(axis=1, ddof=1, keepdims=True)
            mel = (mel - m) / (s + 1e-5)
        return mel.astype(np.float32)


def inspect_ts(path: pathlib.Path) -> None:
    """TorchScript archives keep readable source under code/ — no torch needed."""
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        return
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        print(f"--- {path.name}: {len(names)} entries ---")
        for n in names:
            if n.endswith(".py") or "constants" in n:
                print(f"\n>>> {n}")
                try:
                    src = z.read(n).decode("utf-8", "replace")
                except Exception:  # noqa: BLE001
                    continue
                print(src[:6000])
        blob = b" ".join(z.read(n) for n in names if n.endswith(".py"))
        txt = blob.decode("utf-8", "replace")
        for key in ("n_fft", "hop_length", "win_length", "n_mels", "preemph",
                    "normalize", "log_zero_guard", "mag_power", "sample_rate",
                    "dither", "center", "pad_to"):
            hits = set(re.findall(rf"{key}\s*[:=]\s*([\w\.\-\+]+)", txt))
            if hits:
                print(f"  {key}: {sorted(hits)}")


def dump_golden(out: pathlib.Path, fe: LogMelFrontend) -> None:
    """Deterministic inputs + expected features, for the Swift unit test."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    sr = fe.p["sample_rate"]
    cases = {
        "sine440": np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32) * 0.5,
        "noise": rng.standard_normal(sr // 2).astype(np.float32) * 0.1,
        "chirp": np.sin(
            2 * np.pi * np.cumsum(np.linspace(100, 4000, sr) / sr)
        ).astype(np.float32) * 0.3,
        "short": np.sin(2 * np.pi * 200 * np.arange(1600) / sr).astype(np.float32) * 0.4,
    }
    manifest = {"params": fe.p, "cases": {}}
    for name, wav in cases.items():
        feats = fe(wav)
        wav.tofile(out / f"{name}.input.f32")
        feats.tofile(out / f"{name}.expected.f32")
        manifest["cases"][name] = {
            "n_samples": int(wav.size),
            "n_mels": int(feats.shape[0]),
            "n_frames": int(feats.shape[1]),
            "mean": float(feats.mean()),
            "std": float(feats.std()),
        }
        print(f"  {name}: {wav.size} samples -> {feats.shape} "
              f"mean={feats.mean():+.4f} std={feats.std():.4f}")

    # the filterbank itself, so Swift can load it instead of rebuilding it
    fe.fb.tofile(out / "mel_filterbank.f32")
    manifest["filterbank_shape"] = list(fe.fb.shape)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote golden vectors -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect-ts", type=pathlib.Path)
    ap.add_argument("--dump-golden", type=pathlib.Path)
    args = ap.parse_args()

    if args.inspect_ts:
        inspect_ts(args.inspect_ts)
    if args.dump_golden:
        dump_golden(args.dump_golden, LogMelFrontend())
    if not args.inspect_ts and not args.dump_golden:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
