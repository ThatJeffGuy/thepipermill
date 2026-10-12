"""Recordings -> training dataset.

Converts every audio file to 22.05 kHz mono, splits it at pauses into 1.5-15 s clips, writes down what was said
with Whisper, and saves  data/<voice>/wavs/*.wav  +  data/<voice>/metadata.csv  (`file.wav|text`).
"""
import subprocess, sys, tempfile
from pathlib import Path
from . import voices
from .config import CFG

SR = 22050
EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mkv", ".webm"}


def _convert(src: Path, dst: Path):
    # mono, 22.05 kHz, gentle high-pass against rumble, loudness normalised
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR),
                    "-af", "highpass=f=60,loudnorm=I=-20:TP=-2:LRA=11", "-sample_fmt", "s16", str(dst)], check=True)


def _segments(y, top_db, min_s, max_s, join_gap_s):
    import librosa
    iv = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=256)
    out, cur = [], None
    for a, b in iv:
        if cur is None: cur = [a, b]
        elif (a - cur[1]) / SR < join_gap_s and (b - cur[0]) / SR <= max_s: cur[1] = b
        else: out.append(cur); cur = [a, b]
    if cur: out.append(cur)
    pad, res = int(0.15 * SR), []
    for a, b in out:
        a, b = max(0, a - pad), min(len(y), b + pad)
        if (b - a) / SR >= min_s: res.append((a, b))
    return res


def run(name, top_db=35.0, min_s=1.5, max_s=15.0, join_gap=0.45, log=print):
    import soundfile as sf, torch
    from transformers import pipeline
    src = voices.source_dir(name)
    files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in EXTS) if src.exists() else []
    if not files: raise SystemExit(f"No audio files found in {src}")
    root = CFG.data / name; wavs = root / "wavs"
    for old in wavs.glob("*.wav"): old.unlink()                      # re-preparing starts clean
    wavs.mkdir(parents=True, exist_ok=True)

    clips, n = [], 0
    with tempfile.TemporaryDirectory() as td:
        for f in files:
            tmp = Path(td) / "c.wav"
            try: _convert(f, tmp)
            except subprocess.CalledProcessError: log(f"skipped (could not read): {f.name}"); continue
            y, _ = sf.read(tmp, dtype="float32")
            for s, e in _segments(y, top_db, min_s, max_s, join_gap):
                n += 1; p = wavs / f"{name}_{n:05d}.wav"
                sf.write(p, y[s:e], SR, subtype="PCM_16"); clips.append((p, (e - s) / SR))
            log(f"{f.name}: {n} clips so far")
    if not clips: raise SystemExit("No speech found. Try a quieter recording or a lower --top-db.")
    log(f"{len(clips)} clips, {sum(d for _, d in clips)/60:.1f} min of speech. Writing it down (Whisper)...")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu": log("WARNING: no GPU found for Whisper; this will be slow.")
    asr = pipeline("automatic-speech-recognition", model=CFG["whisper_model"], device=dev,
                   torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
    lang = {"en-us": "english", "en-gb": "english", "en": "english"}.get(CFG["espeak_voice"].lower())
    gen = {"task": "transcribe", "num_beams": 5}
    if lang: gen["language"] = lang
    rows, dropped, B = [], 0, 8
    for i in range(0, len(clips), B):
        batch = clips[i:i + B]
        audio = [sf.read(p, dtype="float32")[0] for p, _ in batch]
        res = asr([{"raw": x, "sampling_rate": SR} for x in audio], batch_size=B, generate_kwargs=gen)
        for (p, d), r in zip(batch, res):
            t = " ".join(r["text"].split()).replace("|", " ")
            words = len(t.split())
            # Whisper invents text on non-speech: drop empties and impossible speaking rates
            if len(t) < 3 or words / d > 6 or words / d < 0.4: dropped += 1; p.unlink(missing_ok=True); continue
            rows.append(f"{p.name}|{t}")
        log(f"  {min(i + B, len(clips))}/{len(clips)}")
    (root / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    mins = sum(d for p, d in clips if p.exists()) / 60
    log(f"\nDone: {len(rows)} clips ({mins:.1f} min), {dropped} dropped.")
    if mins < 20: log("NOTE: under 20 minutes. Fine-tuning works best with 30-60+ minutes of clean speech.")
    log("Next: check the words, then train.")
