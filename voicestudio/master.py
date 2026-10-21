"""Stage 1 of 2.0: turn the raw recordings into a clean MASTER before anything else happens to them.

Everything after this (finding speakers, cutting clips, writing transcripts, cloning) works on the master, not on the raw files.
Why first: the voice cleaner (Demucs) does its best work on long stretches. Cleaning only short pieces after cutting leaves artefacts
at every piece's edges (dead gaps, odd noises), and the speaker/pause detection has an easier job on clean audio.

What the master chain does, in order (each part can be switched off):
  vocals    Demucs keeps only the voices (removes music, effects, background)
  highpass  removes rumble below 70 Hz
  denoise   gentle FFT noise reduction for steady hiss/hum (off by default: heavy use can smooth a gravelly voice)
  boost     a little presence and light compression so quiet words are not lost (off by default)
  loudness  one consistent volume (about -20 LUFS) with a peak limit
  mono      always: one channel at 22.05 kHz
Before a long run, `preview` renders 3 to 5 varied excerpts original-vs-cleaned for you to approve or deny. Nothing is processed in bulk until
you approve. Your original recordings are never changed.
"""
import json, shutil, subprocess, tempfile
from pathlib import Path
from . import audio, clean, speakers, voices
from .config import CFG
from .progress import Progress

SR = 22050
EXCERPT_S = 8.0
DEFAULTS = {"vocals": True, "highpass": True, "denoise": False, "boost": False, "loudness": True, "target_lufs": -20}


def options(name, override=None):
    o = dict(DEFAULTS); o.update(voices.read(name).get("clean_opts") or {}); o.update(override or {}); return o


def chain(o):
    """The ffmpeg filters for the non-AI parts of the chain."""
    f = []
    if o.get("highpass"): f.append("highpass=f=70")
    if o.get("denoise"): f.append("afftdn=nf=-25")
    if o.get("boost"): f += ["equalizer=f=3000:t=q:w=1:g=2", "acompressor=threshold=-21dB:ratio=2:attack=20:release=250:makeup=2"]
    if o.get("loudness"): f.append(f"loudnorm=I={o.get('target_lufs', -20)}:TP=-2:LRA=11")
    return ",".join(f)


def _ffmpeg(src, dst, o):
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR)]
    if chain(o): cmd += ["-af", chain(o)]
    cmd += ["-sample_fmt", "s16", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not Path(dst).exists(): raise RuntimeError("ffmpeg failed: " + (r.stderr.strip().splitlines() or ["?"])[-1][:200])


def process_array(y, o, sep):
    """Float mono audio at SR -> cleaned float mono audio at SR (same chain as a whole file)."""
    import soundfile as sf
    if o.get("vocals") and sep is not None: y = sep.vocals(y)
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "in.wav", Path(td) / "out.wav"
        sf.write(a, y, SR, subtype="FLOAT"); _ffmpeg(a, b, o)
        out, _ = sf.read(b, dtype="float32")
    return out


def master_dir(name): return CFG.runs / name / "master"


def run(name, log=print, override=None):
    """Clean every recording into runs/<name>/master/. Requires the preview to be approved (or skipped with override={'skip_approval': True})."""
    import soundfile as sf
    o = options(name, override)
    if not (override or {}).get("skip_approval") and not approved(name): raise SystemExit("Preview the clean-up and approve it first.")
    files = speakers.audio_files(name)
    if not files: raise SystemExit("No audio files found. Choose your recordings folder first.")
    out = master_dir(name); shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    prog = Progress(name, "master"); sep = clean.Separator(log) if o.get("vocals") else None; done = []
    try:
        for i, f in enumerate(files, 1):
            key = speakers.rel(name, f); prog.update("Cleaning the recordings", i - 1, len(files), key, frac=(i - 1) / len(files))
            try:
                with tempfile.TemporaryDirectory() as td:
                    raw = Path(td) / "raw.wav"; secs = audio.decode(f, raw, SR, log=log); y, _ = sf.read(raw, dtype="float32")
                y2 = process_array(y, o, sep)
                dst = out / (key.replace("/", "__").rsplit(".", 1)[0] + ".wav"); sf.write(dst, y2, SR, subtype="PCM_16")
                done.append({"source": key, "master": dst.name, "minutes": round(secs / 60, 1)}); log(f"[{i}/{len(files)}] {key}: {secs / 60:.1f} min cleaned")
            except (audio.AudioError, RuntimeError, OSError) as e: log(f"[{i}/{len(files)}] SKIPPED {key}: {e}")
    finally:
        if sep: sep.close()
    if not done: raise SystemExit("No recording could be cleaned.")
    (out / "master.json").write_text(json.dumps({"options": o, "files": done}))
    voices.write(name, clean_opts=o, master=True)
    prog.finish(f"{len(done)} file(s)"); log(f"\nDone: {len(done)} cleaned recording(s) in {out}. Next steps use these.")
    return done


# ---------------------------------------------------------------- preview + approval
def master_path(name, f):
    """The cleaned master of a recording (the raw file `f`), or None if it has not been made."""
    p = master_dir(name) / (speakers.rel(name, f).replace("/", "__").rsplit(".", 1)[0] + ".wav")
    return p if p.exists() else None


def available(name):
    """True when a master exists for every recording, so Prepare can use it instead of cleaning short pieces."""
    files = speakers.audio_files(name)
    return bool(files) and (master_dir(name) / "master.json").exists() and all(master_path(name, f) for f in files)


def preview_dir(name): return CFG.runs / name / "clean-preview"


def _pick_excerpts(y, n):
    """Start times (s) of n varied windows: quietest speech, loudest, in between, and spread through the file."""
    import numpy as np
    w = int(EXCERPT_S * SR); hop = w // 2
    if len(y) <= w: return [0.0]
    starts = list(range(0, len(y) - w, hop)); lv = np.array([float(np.sqrt((y[s:s + w] ** 2).mean())) for s in starts])
    ok = [i for i in range(len(starts)) if lv[i] > 0.4 * np.median(lv) and lv[i] > 0.003] or list(range(len(starts)))
    ok.sort(key=lambda i: lv[i]); pick = [ok[0], ok[-1]]
    for q in np.linspace(0.25, 0.75, max(0, n - 2)): pick.append(ok[int(q * (len(ok) - 1))])
    return sorted({round(starts[i] / SR, 1) for i in pick})[:n]


def preview(name, n=4, log=print, override=None):
    """Render n varied excerpts from the recordings, original next to cleaned, for the user to approve or deny."""
    import soundfile as sf
    n = max(3, min(5, n)); o = options(name, override)
    files = speakers.audio_files(name)
    if not files: raise SystemExit("No audio files found. Choose your recordings folder first.")
    d = preview_dir(name); shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True); prog = Progress(name, "cleanprev"); prog.update("Making the previews", 0, n, frac=0.02)
    # spread the excerpts over up to n files (longest first), or over one long file
    sizes = sorted(files, key=lambda p: -p.stat().st_size)[:n]; per = [n // len(sizes) + (1 if i < n % len(sizes) else 0) for i in range(len(sizes))]
    sep = clean.Separator(log) if o.get("vocals") else None; items = []; pad = int(clean.CONTEXT_S * SR)
    try:
        for f, k in zip(sizes, per):
            with tempfile.TemporaryDirectory() as td:
                raw = Path(td) / "raw.wav"; audio.decode(f, raw, SR, log=log); y, _ = sf.read(raw, dtype="float32")
            for t in _pick_excerpts(y, k):
                a = int(t * SR); b = min(len(y), a + int(EXCERPT_S * SR)); c0, c1 = max(0, a - pad), min(len(y), b + pad)
                cleaned = process_array(y[c0:c1], o, sep)[a - c0: a - c0 + (b - a)]
                i = len(items) + 1
                sf.write(d / f"p{i}_original.wav", y[a:b], SR, subtype="PCM_16"); sf.write(d / f"p{i}_cleaned.wav", cleaned, SR, subtype="PCM_16")
                prog.update("Making the previews", i, n, frac=min(0.95, i / n))
                items.append({"id": i, "source": speakers.rel(name, f), "start": t, "seconds": round((b - a) / SR, 1), "approved": None}); log(f"Preview {i}: {speakers.rel(name, f)} at {t:.0f} s")
    finally:
        if sep: sep.close()
    (d / "preview.json").write_text(json.dumps({"options": o, "items": items})); prog.finish(f"{len(items)} previews")
    log("Listen to each pair, then approve or deny it. Denied? Change the options and preview again.")
    return items


def state(name):
    try: return json.loads((preview_dir(name) / "preview.json").read_text())
    except (OSError, ValueError): return None


def decide(name, item_id, ok):
    s = state(name)
    if not s: raise SystemExit("There is no preview yet.")
    hit = [i for i in s["items"] if i["id"] == int(item_id)]
    if not hit: raise SystemExit("Unknown preview clip.")
    hit[0]["approved"] = bool(ok); (preview_dir(name) / "preview.json").write_text(json.dumps(s)); return s


def approved(name):
    """True when every preview clip is approved and the preview was made with the options now in use."""
    s = state(name)
    return bool(s) and all(i["approved"] is True for i in s["items"]) and s["options"] == options(name)
