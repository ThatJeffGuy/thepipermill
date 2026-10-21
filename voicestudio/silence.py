"""Trim silence from the clips.

Every clip keeps a short, natural bit of silence at each end (default 0.15 s before the first word, 0.25 s after the last) and loses the rest.
Optionally, long pauses INSIDE a clip are shortened to a maximum (default 0.6 s) so the voice does not learn to hesitate for seconds.
Words are never touched: silence is found from the loudness of the clip itself, only quiet stretches are shortened, and edges are cross-faded.
The untouched originals are kept once in `wavs.presilence/`. Running it again is harmless.
"""
import shutil
import numpy as np
from . import prep
from .config import CFG
from .progress import Progress

HOP_S, WIN_S, FLOOR_DB, MIN_KEEP_S = 0.01, 0.02, 35.0, 1.0       # loudness frames; silence = more than FLOOR_DB below the clip's loudest frame; never leave less than this


def _silent(y, sr):
    import librosa
    hop, win = int(sr * HOP_S), int(sr * WIN_S)
    db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=win, hop_length=hop)[0] + 1e-6)
    return db < db.max() - FLOOR_DB, hop


def trim_clip(y, sr, lead=0.15, tail=0.25, max_pause=None):
    """-> (new audio, seconds removed). Pure function of one clip."""
    sil, hop = _silent(y, sr); voiced = np.where(~sil)[0]
    if len(voiced) < 3: return y, 0.0
    a = max(0, voiced[0] * hop - int(lead * sr)); b = min(len(y), (voiced[-1] + 1) * hop + int(tail * sr))
    if b - a < MIN_KEEP_S * sr: return y, 0.0
    out = y[a:b]
    if max_pause:                                              # shorten pauses inside the clip, keeping the first and last half of each at full natural length
        sil2, hop2 = _silent(out, sr); runs, i = [], 0         # (positions below are in samples; hop2 is samples per loudness frame)
        while i < len(sil2):
            if sil2[i]:
                k = i
                while k < len(sil2) and sil2[k]: k += 1
                if (k - i) * hop2 > (max_pause + 0.1) * sr and i > 0 and k < len(sil2): runs.append((i * hop2, k * hop2))      # interior pauses only
                i = k
            else: i += 1
        if runs:
            keep, xf, parts, pos = int(max_pause * sr / 2), int(0.01 * sr), [], 0
            for s0, s1 in runs:
                parts.append(out[pos:s0 + keep]); pos = s1 - keep
            parts.append(out[pos:]); parts = [q for q in parts if len(q)]; res = parts[0]
            for q in parts[1:]:                                # a 10 ms cross-fade at every join
                n = min(xf, len(res), len(q))
                if n < 2: res = np.concatenate([res, q]); continue
                f = np.linspace(0, 1, n, dtype="float32")
                res = np.concatenate([res[:-n], res[-n:] * (1 - f) + q[:n] * f, q[n:]])
            out = res
    return out, (len(y) - len(out)) / sr


def run(name, lead=0.15, tail=0.25, pauses=False, max_pause=0.6, log=print):
    import soundfile as sf
    root = CFG.data / name; wavs = root / "wavs"; back = root / "wavs.presilence"
    files = sorted(wavs.glob("*.wav"))
    if not files: raise SystemExit("There are no clips to trim.")
    back.mkdir(exist_ok=True); prog = Progress(name, "silence"); changed = 0; removed = 0.0
    for i, p in enumerate(files, 1):
        prog.update("Trimming silence", i - 1, len(files), p.name, frac=(i - 1) / len(files))
        y, sr = sf.read(p, dtype="float32"); y = y if y.ndim == 1 else y.mean(1)
        out, sec = trim_clip(y, sr, lead, tail, max_pause if pauses else None)
        if sec < 0.02: continue
        if not (back / p.name).exists(): shutil.copy2(p, back / p.name)
        sf.write(p, prep.level(out), sr, subtype="PCM_16"); changed += 1; removed += sec
    shutil.rmtree(CFG.cache / name, ignore_errors=True)                       # Piper's cache is keyed by clip+text
    if changed: (CFG.runs / name / "clipscores.json").unlink(missing_ok=True)
    log(f"\nDone: {changed} of {len(files)} clips trimmed, {removed:.0f} s of silence removed ({removed / max(1, sum(sf.info(f).duration for f in files)) * 100:.1f}% of the audio)."
        + (f" Pauses inside clips are now at most {max_pause:g} s." if pauses else ""))
    log("The untouched originals are in wavs.presilence.")
    prog.finish(f"{changed} clips trimmed")
