"""Stage 1 of 2.0: turn the raw recordings into a clean MASTER before anything else happens to them.

Everything after this (finding speakers, cutting clips, writing transcripts, cloning) works on the master, not on the raw files.
Why first: the voice cleaner (Demucs) does its best work on long stretches. Cleaning only short pieces after cutting leaves artefacts
at every piece's edges (dead gaps, odd noises), and the speaker/pause detection has an easier job on clean audio.

What the master chain does, in order (each part can be switched off):
  vocals    Demucs keeps only the voices (removes music, effects, background)
  highpass  removes rumble below 70 Hz
  denoise   FFT noise reduction for steady hiss and hum
  boost     a little presence and light compression so quiet words are not lost
  loudness  one consistent volume (about -20 LUFS) with a peak limit
  mono      always: one channel at 22.05 kHz
Every step after it (speaker finding, matching, clip cutting, transcripts) reads the master, never the raw files. Your original recordings are never changed.
"""
import json, shutil, subprocess, tempfile
from pathlib import Path
from . import audio, clean, speakers, voices
from .config import CFG
from .progress import Progress

SR = 22050
DEFAULTS = {"vocals": True, "highpass": True, "denoise": True, "boost": True, "loudness": True, "target_lufs": -20}


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
    """Clean every recording into runs/<name>/master/."""
    import soundfile as sf
    o = options(name, override)
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
    """True when a master exists for every recording: every later step then reads the master instead of the raw files."""
    files = speakers.audio_files(name)
    return bool(files) and (master_dir(name) / "master.json").exists() and all(master_path(name, f) for f in files)


def info(name):
    """{'files': n, 'minutes': m} of the finished master, else None."""
    try: j = json.loads((master_dir(name) / "master.json").read_text())
    except (OSError, ValueError): return None
    return {"files": len(j["files"]), "minutes": round(sum(f.get("minutes", 0) for f in j["files"]), 1)}


# ---------------------------------------------------------------- keep / reuse the cleaned recordings
def export_zip(name, fileobj):
    """Write the finished master (cleaned wavs + master.json) to `fileobj` as a zip, so cleaning never has to be run again for these recordings."""
    import zipfile
    d = master_dir(name)
    if not (d / "master.json").exists(): raise ValueError("The recordings have not been cleaned yet.")
    with zipfile.ZipFile(fileobj, "w", zipfile.ZIP_STORED) as z:
        z.write(d / "master.json", "master.json")
        for f in sorted(d.glob("*.wav")): z.write(f, f"master/{f.name}")
        z.writestr("README.txt", "Cleaned recordings made with The Piper Mill (the Clean step).\n\nmaster/*.wav  mono, 16-bit, 22.05 kHz, voices only, one even volume\n"
                   "master.json   which original recording each file came from, and the settings used\n\n"
                   "In The Piper Mill: make a voice with the SAME recordings folder, then on the Clean step press 'Use cleaned recordings' and pick this zip.\n")


def restore_zip(name, path):
    """Install an exported master zip as this voice's cleaned recordings (the recordings folder must be the same one the zip was made from)."""
    import zipfile
    zf = zipfile.ZipFile(path); names = set(zf.namelist())
    if "master.json" not in names: raise ValueError("That zip is not an export of cleaned recordings (no master.json).")
    j = json.loads(zf.read("master.json")); out = master_dir(name)
    files = speakers.audio_files(name)
    keys = {speakers.rel(name, f).replace("/", "__").rsplit(".", 1)[0] + ".wav" for f in files}
    have = {f["master"] for f in j["files"] if f"master/{f['master']}" in names}
    missing = keys - have
    if not files: raise ValueError("Choose the recordings folder first.")
    if missing: raise ValueError(f"This export does not match the recordings folder: {len(missing)} of {len(keys)} recordings have no cleaned copy in it (for example {sorted(missing)[0]}).")
    shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    for n in have:
        with zf.open(f"master/{n}") as src, open(out / n, "wb") as dst: shutil.copyfileobj(src, dst)
    (out / "master.json").write_text(json.dumps(j))
    voices.write(name, clean_opts=j.get("options") or {}, master=True)
    return len(have)


# ---------------------------------------------------------------- listen before you clean
PV_SECONDS = 20
STEPS = ("vocals", "highpass", "denoise", "boost", "loudness")


def preview_dir(name): return CFG.runs / name / "clean-preview"


def preview_info(name):
    try: return json.loads((preview_dir(name) / "index.json").read_text())
    except (OSError, ValueError): return None


def _spot(f, start, secs):
    import io, soundfile as sf
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{start:.2f}", "-t", str(secs), "-i", str(f), "-map", "0:a:0", "-ac", "1", "-ar", str(SR), "-f", "wav", "-"], capture_output=True)
    if r.returncode != 0 or not r.stdout: return None
    y, _ = sf.read(io.BytesIO(r.stdout), dtype="float32"); return y


def sanitize(d):
    """Only the known cleaning switches from a request, as clean values."""
    o = {k: bool(d[k]) for k in STEPS if k in d}
    if "target_lufs" in d: o["target_lufs"] = max(-30.0, min(-14.0, float(d["target_lufs"])))
    return o


def excerpt(name, sample):
    """One random 15-30 s clip of real speech from the raw recordings. The same `sample` number always gives the same clip; the next number gives a different one."""
    import random, numpy as np
    files = speakers.audio_files(name)
    if not files: raise SystemExit("No audio files found. Choose your recordings folder first.")
    rng = random.Random(f"{name}:{sample}"); f = rng.choice(files); d = audio.duration(f); best = None
    for _ in range(6):
        secs = rng.uniform(15, 30); start = rng.uniform(0, max(0.0, d - secs)) if d > secs else 0.0
        y = _spot(f, start, secs)
        if y is None or not len(y): continue
        rms = float(np.sqrt(np.mean(y ** 2)))
        if best is None or rms > best[0]: best = (rms, y)
        if rms > 0.01: break                         # loud enough to be speech, not a gap
    if best is None: raise SystemExit(f"Could not read a sample from {speakers.rel(name, f)}.")
    return speakers.rel(name, f), best[1]


def preview_variants(name, mode, opts=None):
    cur = {**options(name), **(opts or {})}; t = cur.get("target_lufs", -20)
    only = lambda *k: {**{s: False for s in STEPS}, **{s: True for s in k}, "target_lufs": t}
    mine = ("Your settings", cur)
    if mode == "current": return [mine]
    return [("Voices only", only("vocals")), ("Hiss and rumble only", only("highpass", "denoise")),
            ("Presence and even volume only", only("boost", "loudness")), ("Everything", only(*STEPS)), mine]


def preview_run(name, mode="current", opts=None, log=print):
    """Cut a ~20 s sample and clean it with the chosen settings (or with each part and some combinations) so they can be compared by ear."""
    import re, time, soundfile as sf
    d = preview_dir(name); d.mkdir(parents=True, exist_ok=True)
    sample = int(voices.read(name).get("clean_sample") or 0); info = preview_info(name) or {}
    if info.get("sample") != sample: [p.unlink() for p in d.glob("*")]; info = {}
    key, y = excerpt(name, sample); items = [i for i in info.get("items", [])]
    variants = preview_variants(name, mode, opts); prog = Progress(name, "cleanpreview")
    slug = lambda s: re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    def put(label, arr, o):
        for p in d.glob(slug(label) + "-*.wav"): p.unlink()
        fn = f"{slug(label)}-{int(time.time() * 1000) % 10**9}.wav"; sf.write(d / fn, arr, SR, subtype="PCM_16")
        items[:] = [i for i in items if i["label"] != label] + [{"label": label, "file": fn, "opts": o}]
    if not any(i["label"] == "Original" for i in items): put("Original", y, None)
    sep = clean.Separator(log) if any(o.get("vocals") for _, o in variants) else None
    try:
        for k, (label, o) in enumerate(variants, 1):
            prog.update("Cleaning the sample", k - 1, len(variants), label, frac=(k - 1) / len(variants))
            put(label, process_array(y, o, sep), o); log(f"{label}: done")
    finally:
        if sep: sep.close()
    order = ["Original", "Voices only", "Hiss and rumble only", "Presence and even volume only", "Everything", "Your settings"]
    items.sort(key=lambda i: order.index(i["label"]) if i["label"] in order else 99)
    (d / "index.json").write_text(json.dumps({"sample": sample, "source": key, "seconds": round(len(y) / SR), "items": items}))
    prog.finish(f"{len(variants)} sample(s)")
