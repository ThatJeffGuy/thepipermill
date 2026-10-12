"""Recordings -> training dataset.

Reads every audio/video file in your recordings folder, splits it at pauses into clips (never longer than 15 s), writes down
what was said with Whisper, and saves  data/<voice>/wavs/*.wav  +  data/<voice>/metadata.csv  (`file.wav|text`).

Built for real recordings: long files, many files, noisy rooms, stereo/surround, odd sample rates, and files that cannot be read
(those are skipped and listed, they never stop the run).
"""
import tempfile
import numpy as np
from pathlib import Path
from . import audio, speakers, voices
from .config import CFG
from .progress import Progress

SR = 22050
HOP = int(0.02 * SR)                       # 20 ms frames
# things Whisper "hears" in silence or noise
JUNK = {"thank you.", "thanks for watching.", "thank you for watching.", "you", "bye.", "so", "okay.", "uh", "um", "mm-hmm."}


def _frame_db(y):
    import librosa
    r = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    return 20 * np.log10(r + 1e-9)


def analyze(y):
    """(top_db to use for this recording, signal-to-noise estimate in dB). A noisy room needs a stricter silence threshold."""
    db = _frame_db(y)
    if len(db) < 10: return 35.0, 99.0
    peak, floor = np.percentile(db, 99.5), np.percentile(db, 15)
    return float(np.clip(peak - (floor + 8), 20, 45)), float(np.percentile(db, 90) - floor)


def _enforce_max(y, a, b, max_s):
    """Split a stretch longer than max_s at the quietest moments, so no clip is ever too long."""
    max_n, min_n, out = int(max_s * SR), int(max_s * 0.5 * SR), []
    while b - a > max_n:
        lo, hi = a + min_n, a + max_n
        n = (hi - lo) // HOP
        if n < 2: cut = hi
        else:
            seg = y[lo:lo + n * HOP].reshape(n, HOP)
            cut = lo + int(np.argmin((seg ** 2).mean(axis=1))) * HOP + HOP // 2
        out.append((a, cut)); a = cut
    out.append((a, b))
    return out


def segments(y, top_db, min_s, max_s, join_gap_s):
    """Phrase-sized clips (min_s..max_s seconds) from one stretch of audio."""
    import librosa
    iv = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=256)
    merged, cur = [], None
    for a, b in iv:
        if cur is None: cur = [a, b]
        elif (a - cur[1]) / SR < join_gap_s and (b - cur[0]) / SR <= max_s: cur[1] = b
        else: merged.append(cur); cur = [a, b]
    if cur: merged.append(cur)
    pad, res = int(0.15 * SR), []
    for a, b in merged:
        a, b = max(0, a - pad), min(len(y), b + pad)
        for s, e in _enforce_max(y, a, b, max_s):
            if (e - s) / SR >= min_s and float(np.sqrt((y[s:e] ** 2).mean())) > 0.003: res.append((s, e))   # skip near-silence
    return res


def _transcribe(asr, audio_list, gen, batch):
    """Run Whisper; if the GPU runs out of memory, retry with smaller batches."""
    import torch
    while True:
        try:
            return asr([{"raw": x, "sampling_rate": SR} for x in audio_list], batch_size=batch, generate_kwargs=gen), batch
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch == 1: raise
            batch = max(1, batch // 2)


def run(name, top_db=None, min_s=1.5, max_s=15.0, join_gap=0.45, log=print):
    import soundfile as sf, torch
    from transformers import pipeline
    src = voices.source_dir(name)
    files = speakers.audio_files(name)
    if not files: raise SystemExit(f"No audio files found in {src or 'a folder (choose your recordings folder first)'}")
    chosen = voices.read(name).get("keep_people")
    if chosen:
        if speakers.stale(name): log("NOTE: the recordings changed since the speakers were found; press 'Find the speakers' again.")
        log(f"Using only the parts where person(s) {', '.join(map(str, chosen))} speak.")
    prog = Progress(name, "prep")
    root = CFG.data / name; wavs = root / "wavs"
    for old in wavs.glob("*.wav"): old.unlink()                      # re-preparing starts clean
    wavs.mkdir(parents=True, exist_ok=True)

    clips, n, skipped, noisy, silent = [], 0, [], [], []
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(files, 1):
            key = speakers.rel(name, f)
            prog.update("Cutting the recordings into clips", i - 1, len(files), key)
            tmp = Path(td) / "c.wav"
            try:
                secs = audio.decode(f, tmp, SR, loudnorm=True, log=log)
                y, _ = sf.read(tmp, dtype="float32")
                iv = speakers.intervals(name, key)                    # None = no one chosen: use everything
                spans = [(0, len(y))] if iv is None else [(int(a * SR), min(len(y), int(b * SR))) for a, b in iv]
                got = 0
                for s0, e0 in spans:
                    part = y[s0:e0]
                    if len(part) < min_s * SR: continue
                    db, snr = (top_db, None) if top_db else analyze(part)
                    new = segments(part, db, min_s, max_s, join_gap)
                    if new and snr is not None and snr < 18: noisy.append(key)
                    for s, e in new:
                        n += 1; got += 1; p = wavs / f"{name}_{n:05d}.wav"
                        sf.write(p, part[s:e], SR, subtype="PCM_16"); clips.append((p, (e - s) / SR))
                if got == 0: silent.append(key)
                log(f"[{i}/{len(files)}] {key}: {secs/60:.1f} min -> {got} clips" + ("  (no speech found)" if got == 0 else ""))
            except (audio.AudioError, RuntimeError, ValueError, OSError) as e:
                skipped.append((key, str(e))); log(f"[{i}/{len(files)}] SKIPPED {key}: {e}")
            finally:
                tmp.unlink(missing_ok=True)
    if not clips:
        raise SystemExit("No speech found in any file." + (f" {len(skipped)} file(s) could not be read." if skipped else " Try a quieter recording."))
    log(f"\n{len(clips)} clips, {sum(d for _, d in clips)/60:.1f} min of speech from {len(files) - len(skipped)} file(s). Writing it down (Whisper)...")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu": log("WARNING: no GPU found for Whisper; this will be slow.")
    asr = pipeline("automatic-speech-recognition", model=CFG["whisper_model"], device=dev,
                   torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
    lang = {"en-us": "english", "en-gb": "english", "en": "english"}.get(CFG["espeak_voice"].lower())
    gen = {"task": "transcribe", "num_beams": 5, **({"language": lang} if lang else {})}
    rows, dropped, batch, done = [], 0, 8, 0
    while done < len(clips):
        chunk = clips[done:done + batch]
        prog.update("Writing down what was said", done, len(clips), f"{done}/{len(clips)} clips")
        res, batch = _transcribe(asr, [sf.read(p, dtype="float32")[0] for p, _ in chunk], gen, batch)
        for (p, d), r in zip(chunk, res):
            t = " ".join(r["text"].split()).replace("|", " ")
            words = len(t.split())
            # Whisper invents text on non-speech: drop empties, known junk and impossible speaking rates
            if len(t) < 3 or t.lower() in JUNK or words / d > 6 or words / d < 0.4: dropped += 1; p.unlink(missing_ok=True); continue
            rows.append(f"{p.name}|{t}")
        done += len(chunk)
        if done % 80 < len(chunk): log(f"  {done}/{len(clips)}")
    (root / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    mins = sum(d for p, d in clips if p.exists()) / 60
    log(f"\nDone: {len(rows)} clips ({mins:.1f} min), {dropped} dropped.")
    if skipped:
        log(f"\n{len(skipped)} file(s) were skipped:"); [log(f"  - {k}: {m}") for k, m in skipped[:25]]
        if len(skipped) > 25: log(f"  ... and {len(skipped) - 25} more")
    if silent:
        log(f"\n{len(silent)} file(s) had no usable speech: {', '.join(silent[:6])}{' ...' if len(silent) > 6 else ''}")
    if noisy:
        u = sorted(set(noisy)); log(f"\nNOTE: {len(u)} recording(s) look noisy (background noise close to the speech level), e.g. {u[0]}. Cleaner audio gives a clearer voice.")
    if mins < 20: log("NOTE: under 20 minutes. Fine-tuning works best with 30-60+ minutes of clean speech.")
    log("Next: check the words, then train.")
    prog.finish(f"{len(rows)} clips ready")
