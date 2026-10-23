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
