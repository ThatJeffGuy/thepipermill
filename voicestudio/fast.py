"""The fast path (2.0): a short reference clip is cloned into a clean dataset, which the normal Train step then fine-tunes.

1. `reference`  - choose one clean clip (about 5 to 12 seconds) and, if you like, type what is said; otherwise Whisper writes it down.
2. `run`        - the cloner (OmniVoice, or F5-TTS if `clone_engine` says so; in its own environment) speaks a few hundred varied sentences in that voice, Whisper checks the
                  words, a voiceprint check drops clips that drifted from the reference, and the best minutes are kept.
3. The kept clips are written exactly like an imported dataset (22.05 kHz mono wavs + metadata.csv), so Check, Train and Listen follow unchanged.

Only record or clone voices of people who have agreed to it.
"""
import json, random, shutil, subprocess, sys, time
from pathlib import Path
from . import audio, textnorm, voices
from .config import CFG
from .progress import Progress

ROOT = Path(__file__).resolve().parent.parent
import os
CLONE_PY = Path(os.environ.get("VOICESTUDIO_CLONE_PY") or ROOT / ".venv-clone" / "bin" / "python")      # (the override is for testing)
WORKER = Path(__file__).with_name("fast_worker.py")
CORPUS = Path(__file__).with_name("data") / "sentences_en.txt"
SR = 22050
REF_MIN_S, REF_MAX_S = 3.0, 12.0


_AV = (0.0, None)


def available():
    """(ok, message). The cloner lives in its own environment so it can never disturb training. Cached for a minute (the check starts Python)."""
    global _AV
    if _AV[1] is not None and time.time() - _AV[0] < 60: return _AV[1]
    if not CLONE_PY.exists(): res = (False, "The fast path is not installed. Run  scripts/install-clone.sh  once (it downloads about 6 GB), then reload this page.")
    else:
        r = subprocess.run([str(CLONE_PY), "-c", "import sys, types; m = types.ModuleType('datasets'); m.Dataset = object; m.load_from_disk = None; sys.modules['datasets'] = m; import omnivoice, f5_tts, speechbrain"], capture_output=True, text=True)
        res = (r.returncode == 0, "" if r.returncode == 0 else "The clone environment is damaged. Run  scripts/install-clone.sh  again.")
    _AV = (time.time(), res); return res


def dir_for(name):
    d = CFG.runs / name / "fast"; d.mkdir(parents=True, exist_ok=True); return d


def _transcribe(wav):
    import soundfile as sf, torch
    from transformers import pipeline
    dev = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=CFG["whisper_model"], device=dev, torch_dtype=torch.float16 if dev == 0 else torch.float32)
    a, sr = sf.read(wav, dtype="float32"); a = a if a.ndim == 1 else a.mean(1)
    lang = (CFG["espeak_voice"] or "en").split("-")[0]
    try: return asr({"raw": a, "sampling_rate": sr}, generate_kwargs={"task": "transcribe", "language": lang, "num_beams": 5})["text"].strip()
    except ValueError: return asr({"raw": a, "sampling_rate": sr}, generate_kwargs={"num_beams": 5})["text"].strip()      # English-only model


def _cut_to_pause(wav, max_s):
    """A reference longer than max_s is cut at the last pause that fits, so it never ends mid-word."""
    import librosa, soundfile as sf
    y, sr = sf.read(wav, dtype="float32")
    if len(y) / sr <= max_s: return
    end = max_s
    for a, b in librosa.effects.split(y, top_db=35, frame_length=2048, hop_length=256):
        if b / sr <= max_s: end = b / sr
    sf.write(wav, y[: int(min(len(y), (end + 0.12) * sr))], sr, subtype="PCM_16")


def reference(name, src, text=None, log=print):
    """Save `src` (any audio/video file) as this voice's reference clip. Returns (seconds, text). Whisper writes the text if none is given."""
    src = Path(src).expanduser()
    if not src.is_file(): raise SystemExit(f"Not a file: {src}")
    out = dir_for(name) / "ref.wav"
    try: audio.decode(src, out, 24000, log=log)
    except audio.AudioError as e: raise SystemExit(str(e))
    _cut_to_pause(out, REF_MAX_S)
    import soundfile as sf
    secs = sf.info(out).duration
    if secs < REF_MIN_S: raise SystemExit(f"That clip is only {secs:.1f} s. Use a clean clip of {REF_MIN_S:.0f} to {REF_MAX_S:.0f} seconds with one person speaking.")
    text = (text or "").strip()
    if not text:
        log("Writing down what is said in the clip...")
        text = _transcribe(out)
    voices.write(name, fast=True, fast_ref_text=text, fast_ref_source=str(src))
    log(f"Reference: {secs:.1f} s. \"{text}\"")
    return secs, text


def sentences(name):
    """The sentences to speak: the bundled set plus an optional file of your own lines (one per line), in a fixed shuffled order.
    If you tick 'only my lines' the bundled set is not used."""
    m = voices.read(name); out = []
    own = m.get("fast_text_file")
    if own and Path(own).expanduser().is_file(): out += [l.strip() for l in Path(own).expanduser().read_text(encoding="utf-8", errors="replace").splitlines() if len(l.strip()) >= 8]
    if not (own and m.get("fast_text_only")): out += [l.strip() for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    seen, uniq = set(), []
    for l in out:
        if l not in seen: seen.add(l); uniq.append(l)
    random.Random(1).shuffle(uniq)
    return uniq


def run(name, minutes=None, log=print, device=None):
    m = voices.read(name)
    ok, why = available()
    if not ok: raise SystemExit(why)
    ref = dir_for(name) / "ref.wav"
    if not ref.exists() or not m.get("fast_ref_text"): raise SystemExit("Choose the reference clip first.")
    minutes = float(minutes or m.get("fast_minutes") or 40)
    d = dir_for(name); shutil.rmtree(d / "kept", ignore_errors=True); (d / "report.json").unlink(missing_ok=True)
    import torch
    job = {"ref_wav": str(ref), "ref_text": m["fast_ref_text"], "sentences": sentences(name), "out_dir": str(d), "target_minutes": minutes,
           "device": device or ("cuda" if torch.cuda.is_available() else "cpu"), "gpu_fraction": float(m.get("fast_gpu_fraction") or CFG["clone_gpu_fraction"]),
           "engine": CFG["clone_engine"], "speed": float(m.get("fast_speed") or CFG["clone_speed"]),
           "whisper_model": CFG["whisper_model"], "language": (CFG["espeak_voice"] or "en").split("-")[0], "cache_dir": str(CFG.home / "models")}
    (d / "job.json").write_text(json.dumps(job))
    prog = Progress(name, "fast"); prog.update("Starting the cloner", 0, 1, frac=0.01)
    log(f"Cloning: about {minutes:.0f} minutes of clean speech from {len(job['sentences'])} sentences (device {job['device']}).")
    p = subprocess.Popen([str(CLONE_PY), "-u", str(WORKER), str(d / "job.json")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = 0
    while p.poll() is None:
        time.sleep(2)
        try:
            pg = json.loads((d / "progress.json").read_text())
            if pg["t"] != last and pg["stage"] != "done":
                last = pg["t"]; prog.update(pg["stage"], pg["done"], max(1, pg["total"]), pg.get("note", ""), frac=min(0.97, 0.02 + 0.95 * _overall(pg)))
        except (OSError, ValueError, KeyError): pass
    tail = p.stdout.read()
    if p.returncode != 0 or not (d / "kept.csv").exists(): raise SystemExit("The cloner stopped with an error:\n" + "\n".join(tail.strip().splitlines()[-12:]))
    return _write_dataset(name, d, log, prog)


_STAGES = {"Speaking the sentences in the reference voice": (0.0, 0.7), "Checking the words": (0.7, 0.9), "Checking that it sounds like the reference": (0.9, 1.0)}


def _overall(pg):
    a, b = _STAGES.get(pg["stage"], (0, 1)); return a + (b - a) * min(1.0, pg["done"] / max(1, pg["total"]))


def _write_dataset(name, d, log, prog):
    import librosa, soundfile as sf
    from . import prep
    root = CFG.data / name; wavs = root / "wavs"; shutil.rmtree(root, ignore_errors=True); wavs.mkdir(parents=True)
    rows = [l for l in (d / "kept.csv").read_text(encoding="utf-8").splitlines() if "|" in l]; out = []
    for i, r in enumerate(rows, 1):
        f, _, t = r.partition("|"); y, sr = sf.read(d / "kept" / f, dtype="float32"); y = y if y.ndim == 1 else y.mean(1)
        if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        fn = f"{name}_{i:05d}.wav"; sf.write(wavs / fn, prep.level(y), SR, subtype="PCM_16"); out.append(f"{fn}|{textnorm.normalize(t, CFG['espeak_voice']).replace('|', ' ')}")
    (root / "metadata.csv").write_text("\n".join(out) + "\n", encoding="utf-8"); (root / "metadata.orig.csv").write_text("\n".join(out) + "\n", encoding="utf-8")
    rep = json.loads((d / "report.json").read_text())
    voices.write(name, imported=True, words_seen=True, fast=True, fast_report=rep)
    log(f"\nDone: {rep['kept']} clips ({rep['minutes']} min) from {rep['generated']} generated. Dropped for wrong words: {rep['dropped_words']}, "
        f"for drifting from the reference voice: {rep['dropped_voice']}. Mean similarity to the reference {rep['mean_similarity']}.")
    log("Next: listen to a few clips, then train (a short run is plenty).")
    prog.finish(f"{rep['kept']} clips"); return rep
