"""Recordings -> training dataset.

Reads every audio/video file in your recordings folder, splits it at pauses into clips (never longer than 15 s), writes down
what was said with Whisper, and saves  data/<voice>/wavs/*.wav  +  data/<voice>/metadata.csv  (`file.wav|text`).

Built for real recordings: long files, many files, noisy rooms, stereo/surround, odd sample rates, and files that cannot be read
(those are skipped and listed, they never stop the run).
"""
import json, shutil, tempfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path
from . import audio, clean, dataset, speakers, textnorm, voices
from .config import CFG
from .progress import Progress

SR = 22050
AHEAD = 3                                  # recordings decoded in the background ahead of the one being cut
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
        except ValueError:                                   # an English-only Whisper model accepts no language/task options: try again without
            gen = {k: v for k, v in gen.items() if k not in ("language", "task")}
            return asr([{"raw": x, "sampling_rate": SR} for x in audio_list], batch_size=batch, generate_kwargs=gen), batch
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch == 1: raise
            batch = max(1, batch // 2)


def active_seconds(y, top_db=None):
    """Seconds of a recording that have sound above the room noise: a rough measure of how much there is to use."""
    import librosa
    iv = librosa.effects.split(y, top_db=top_db or analyze(y)[0], frame_length=1024, hop_length=256)
    return float(sum(b - a for a, b in iv)) / SR


def level(clip, target_rms=0.1, peak=0.95, max_gain=4.0):
    """Same loudness for every clip (Piper learns better from consistent levels), without clipping or boosting noise."""
    rms, pk = float(np.sqrt((clip ** 2).mean())) + 1e-9, float(np.abs(clip).max()) + 1e-9
    return (clip * min(target_rms / rms, peak / pk, max_gain)).astype("float32")


def _thin(plan, frac, chunk=300.0):
    """An even spread through everything: keep about `frac` of the speech, in pieces of up to `chunk` seconds, across all files."""
    seen = kept = 0.0; out = {}
    for key, spans in plan.items():
        if spans is None: out[key] = None; continue
        res = []
        for a, b in spans:
            t = a
            while t < b - 1e-6:
                e = min(b, t + chunk); ln = e - t; seen += ln
                if kept + ln <= frac * seen + ln * 0.5:
                    kept += ln
                    if res and abs(res[-1][1] - t) < 1e-6: res[-1] = (res[-1][0], e)
                    else: res.append((t, e))
                t = e
        out[key] = res
    return out


def run(name, top_db=None, min_s=1.5, max_s=12.0, join_gap=0.45, log=print):
    import soundfile as sf, torch
    from transformers import pipeline
    src = voices.source_dir(name)
    files = speakers.audio_files(name)
    if not files: raise SystemExit(f"No audio files found in {src or 'a folder (choose your recordings folder first)'}")
    one = bool(voices.read(name).get("one_person"))                           # "these recordings are one person": use everything, ignore any speaker choices
    chosen = None if (one or voices.read(name).get("use_match")) else voices.read(name).get("keep_people")      # a matched voice replaces the ticked people
    unscanned = set(((speakers.load(name) or {}).get("scan") or {}).get("skipped", []))
    if unscanned and not one and (voices.read(name).get("use_match") or chosen):                                   # the speaker scan used an even sample of the recordings: so does Prepare
        files = [f for f in files if speakers.rel(name, f) not in unscanned]
    if chosen:
        if speakers.stale(name): log("NOTE: the recordings changed since the speakers were found; press 'Find the speakers' again.")
        log(f"Using only the parts where person(s) {', '.join(map(str, chosen))} speak.")
    mm = voices.read(name); voices.write(name, words_seen=False)           # new clips: the words have not been looked at yet
    if one: log("These recordings are one person: using everything in them (the speaker scan is not used).")
    elif mm.get("use_match"): log("Using only the parts that sound like the clips you ticked ('find more like these').")
    prog = Progress(name, "prep")
    root = CFG.data / name; wavs = root / "wavs"
    (CFG.runs / name / "prep.approved.json").unlink(missing_ok=True)      # new clips: earlier approvals no longer apply
    for old in wavs.glob("*.wav"): old.unlink()                      # re-preparing starts clean
    for stale in ("metadata.csv", "metadata.orig.csv", "metadata.train.csv"): (root / stale).unlink(missing_ok=True)      # so a stopped run is never mistaken for a finished dataset
    wavs.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(CFG.cache / name, ignore_errors=True)               # Piper's cache is keyed by row+text: never reuse it for new clips

    lim = dataset.limit(name); budget = lim * speakers.CUSHION * 60
    plan, total = {}, 0.0                                                      # what to prepare, in seconds, per file
    for f in files:
        key = speakers.rel(name, f); iv = speakers.intervals(name, key)
        if iv is None:
            d = audio.duration(f); plan[key] = [(0.0, d)] if d else None
        else: plan[key] = [(a, b) for a, b in iv]
        total += sum(b - a for a, b in plan[key] or [])
    if total > budget:
        plan = _thin(plan, budget / total)
        log(f"There are {total / 60:.0f} min of speech to prepare, more than a voice needs. Preparing an even spread of about {budget / 60:.0f} min "
            f"(your {lim}-minute limit plus some room to drop weak clips). This keeps the later steps fast and small.")
    from . import master
    use_master = master.available(name)
    if use_master: log("Using your cleaned recordings (the 'Clean the audio' step), so nothing is cleaned again here."); sep = None
    else:
        log("Removing music and background noise from the parts that are kept (only the voice is left in)...")
        sep = clean.Separator(log)
    clips, n, skipped, noisy, silent, heard_s = [], 0, [], [], [], 0.0
    def load(i, f, td):                                                  # decode + read one recording (runs ahead on spare cores while the GPU works)
        tmp = Path(td) / f"c{i}.wav"
        try:
            secs = audio.decode(master.master_path(name, f) if use_master else f, tmp, SR, loudnorm=not use_master, log=log)
            y, _ = sf.read(tmp, dtype="float32")
            return secs, y
        finally:
            tmp.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory() as td, ThreadPoolExecutor(max_workers=AHEAD) as pool:
        futs = {}
        for i, f in enumerate(files, 1):
            for j in range(i, min(len(files), i + AHEAD) + 1):                  # keep the next few recordings decoding in the background
                if j not in futs: futs[j] = pool.submit(load, j, files[j - 1], td)
            key = speakers.rel(name, f)
            W1 = 0.35                                                           # share of the whole job spent reading + cutting (the rest is Whisper)
            prog.update("Reading the recording", i - 1, len(files), key, frac=W1 * (i - 1) / len(files))
            try:
                secs, y = futs.pop(i).result()
                try: heard_s += active_seconds(y, top_db)
                except Exception: pass                                  # only used for the "how much did we use" note
                iv = plan.get(key)                                    # None = no one chosen: use everything
                spans = [(0, len(y))] if iv is None else [(int(a * SR), min(len(y), int(b * SR))) for a, b in iv]
                got = 0
                for si, (s0, e0) in enumerate(spans):
                    prog.update("Cutting into clips", i - 1, len(files), key, frac=W1 * ((i - 1) + 0.2 + 0.8 * si / max(1, len(spans))) / len(files))
                    part = y[s0:e0]
                    if len(part) < min_s * SR: continue
                    if sep: part = sep.span(y, s0, e0)                    # vocals only; the clips and their transcripts come from this
                    db, snr = (top_db, None) if top_db else analyze(part)
                    new = segments(part, db, min_s, max_s, join_gap)
                    if new and snr is not None and snr < 18: noisy.append(key)
                    for s, e in new:
                        n += 1; got += 1; p = wavs / f"{name}_{n:05d}.wav"
                        sf.write(p, level(part[s:e]), SR, subtype="PCM_16"); clips.append((p, (e - s) / SR))
                if got == 0: silent.append(key)
                log(f"[{i}/{len(files)}] {key}: {secs/60:.1f} min -> {got} clips" + ("  (no speech found)" if got == 0 else ""))
            except (audio.AudioError, RuntimeError, ValueError, OSError) as e:
                skipped.append((key, str(e))); log(f"[{i}/{len(files)}] SKIPPED {key}: {e}")
    if sep: sep.close()
    if not clips:
        raise SystemExit("No speech found in any file." + (f" {len(skipped)} file(s) could not be read." if skipped else " Try a quieter recording."))
    log(f"\n{len(clips)} clips, {sum(d for _, d in clips)/60:.1f} min of speech from {len(files) - len(skipped)} file(s). Writing it down (Whisper)...")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu": log("WARNING: no GPU found for Whisper; this will be slow.")
    asr = pipeline("automatic-speech-recognition", model=CFG["whisper_model"], device=dev,
                   torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
    lang = {"en-us": "english", "en-gb": "english", "en": "english"}.get(CFG["espeak_voice"].lower())
    gen = {"task": "transcribe", "num_beams": int(CFG["whisper_beams"]), **({"language": lang} if lang else {})}
    rows, dropped, batch, done = [], 0, 16, 0
    while done < len(clips):
        chunk = clips[done:done + batch]
        prog.update("Writing down what was said", done, len(clips), f"{done}/{len(clips)} clips", frac=0.35 + 0.55 * done / len(clips))
        res, batch = _transcribe(asr, [sf.read(p, dtype="float32")[0] for p, _ in chunk], gen, batch)
        for (p, d), r in zip(chunk, res):
            raw = " ".join(r["text"].split())
            t = textnorm.normalize(raw, CFG["espeak_voice"]).replace("|", " ")      # spoken form: what Piper's espeak-ng will read
            words = len(t.split())
            # Whisper invents text on non-speech: drop empties, known junk, repeated phrases and impossible speaking rates
            if len(t) < 3 or t.lower().strip(".!? ") in JUNK or textnorm.looks_like_loop(t) or words / d > 6 or words / d < 0.4:
                dropped += 1; p.unlink(missing_ok=True); continue
            rows.append(f"{p.name}|{t}")
        done += len(chunk)
        if done % 80 < len(chunk): log(f"  {done}/{len(clips)}")
    (root / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (root / "metadata.orig.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")        # as written by Whisper: lets "reprocess" see which clips you edited
    shutil.rmtree(root / "wavs.untrimmed", ignore_errors=True)
    (CFG.runs / name / "clipscores.json").unlink(missing_ok=True)                  # new clips: old scores no longer apply
    del asr; torch.cuda.empty_cache()
    prog.update("Choosing the best clips", 0, 1, frac=0.9)
    _, culled = dataset.cull(name, lim, log, frac_range=(0.9, 1.0), prog=prog)       # keep only the best `lim` minutes in the dataset
    mins = sum(d for p, d in clips if p.exists()) / 60
    limited = bool(chosen) or bool(mm.get("use_match") and not one)
    cov = {"heard_min": round(heard_s / 60, 1), "used_min": round(mins, 1), "limited": limited, "files": len(files)}
    cov["warn"] = bool(limited and heard_s > 0 and mins * 60 < 0.6 * heard_s)
    (CFG.runs / name / "prep.coverage.json").write_text(json.dumps(cov))
    if heard_s > 0:
        log(f"\nThere was about {cov['heard_min']} min of sound in your recordings; {cov['used_min']} min of it became clips ({round(100 * mins * 60 / heard_s)}%).")
        if cov["warn"]: log("Much of it was left out because you chose specific people or 'find more like these'. If your recordings are mostly one person, use the 'use everything and prepare again' button under Prepare.")
    log(f"\nDone: {len(rows) - culled} clips ({mins:.1f} min), {dropped} dropped as non-speech" + (f", {culled} more removed to stay within {lim} minutes." if culled else "."))
    if skipped:
        log(f"\n{len(skipped)} file(s) were skipped:"); [log(f"  - {k}: {m}") for k, m in skipped[:25]]
        if len(skipped) > 25: log(f"  ... and {len(skipped) - 25} more")
    if silent:
        log(f"\n{len(silent)} file(s) had no usable speech: {', '.join(silent[:6])}{' ...' if len(silent) > 6 else ''}")
    if mins < 30: log("NOTE: under 30 minutes. Fine-tuning works best with 60 to 120 minutes of clean speech.")
    log("Next: check the words, then train.")
    prog.finish(f"{len(rows) - culled} clips ready")
