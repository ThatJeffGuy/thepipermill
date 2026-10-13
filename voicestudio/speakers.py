"""Who is speaking?  For recordings with more than one person in them.

1. `run` finds the speakers in every recording (pyannote.audio diarization), then groups the same person across files
   by comparing voice fingerprints, and saves a short sample clip of each person so you can listen and choose.
2. `keep` saves which people to train on.
3. Prepare then uses only the parts where the chosen people speak.

Only use recordings of people who agree to it.

The diarization model (pyannote/speaker-diarization-community-1, CC-BY-4.0) is gated on Hugging Face: accept its
conditions once on its web page and give a read token (config `hf_token`, the HF_TOKEN variable, or `hf auth login`).
"""
import json, os, re, subprocess, time
from pathlib import Path
from . import audio, voices
from .progress import Progress
from .config import CFG

SR = 16000
EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mkv", ".webm"}
MIN_SAMPLE_S, MAX_SAMPLE_S, N_SAMPLES = 1.5, 8.0, 6


def audio_files(name):
    src = voices.source_dir(name)
    return sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in EXTS) if src and src.exists() else []


def rel(name, f): return f.relative_to(voices.source_dir(name)).as_posix()


def result_path(name): return CFG.runs / name / "speakers.json"


def load(name):
    try: return json.loads(result_path(name).read_text())
    except (OSError, ValueError): return None


def _token():
    t = CFG["hf_token"] or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t: return t
    for p in (Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"):
        if p.exists(): return p.read_text().strip()
    return None


def _load_pipeline():
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        raise SystemExit("pyannote.audio is not installed. Run ./install.sh again (or: .venv/bin/pip install pyannote.audio).")
    model = CFG["diarization_model"]
    try:
        pipe = Pipeline.from_pretrained(model, token=_token())
    except Exception:
        raise SystemExit(f"Could not open the speaker model ({model}). It is gated, so once only:\n"
                         f"  1. Log in at https://huggingface.co and open https://huggingface.co/{model}, accept the conditions.\n"
                         "  2. Create a read token at https://huggingface.co/settings/tokens.\n"
                         "  3. Put it in config.toml as  hf_token = \"...\"  (or set HF_TOKEN, or run `.venv/bin/hf auth login`).")
    import torch
    if torch.cuda.is_available(): pipe.to(torch.device("cuda"))
    return pipe


def cluster(embeddings, threshold, files=None):
    """Group voice fingerprints (one per speaker per file) into people. Returns a list of cluster ids.

    `files` says which recording each fingerprint came from. Two speakers the model already told apart inside one recording
    are different people, so they are never merged (a distance of 2 is above any threshold, and average linkage keeps it that way).
    """
    import numpy as np
    if len(embeddings) == 1: return [0]
    from sklearn.cluster import AgglomerativeClustering
    x = np.asarray(embeddings, dtype="float64"); x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    d = np.clip(1 - x @ x.T, 0, 2); np.fill_diagonal(d, 0)
    if files is not None:
        same = np.asarray([[a == b for b in files] for a in files]); d[same & ~np.eye(len(files), dtype=bool)] = 2.0
    lab = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=threshold).fit_predict(d)
    return [int(i) for i in lab]


def _separator(log):
    """The background remover, or None if it cannot be loaded (the samples are then left as they are)."""
    try:
        from . import clean
        return clean.Separator(log)
    except BaseException as e:
        log(f"Could not load the background remover ({e}); samples will include background sound."); return None


def _grab(sep, wav, a, b):
    """Audio of wav[a:b] seconds at SR, with music and background noise removed when `sep` is given."""
    import soundfile as sf
    if sep is None: return sf.read(wav, start=int(a * SR), stop=int(b * SR), dtype="float32")[0]
    import librosa
    from . import clean
    pad = clean.CONTEXT_S; c0 = max(0.0, a - pad)
    y = sf.read(wav, start=int(c0 * SR), stop=int((b + pad) * SR), dtype="float32")[0]
    y = librosa.resample(y, orig_sr=SR, target_sr=clean.SR)
    v = sep.vocals(y); v = librosa.resample(v, orig_sr=clean.SR, target_sr=SR)
    o = int(round((a - c0) * SR)); return v[o:o + int((b - a) * SR)]


def _samples(name, people, per_file, log, sep=None):
    """Write a few short sample clips per person, spread through the recordings in time (not just their longest turns),
    so what you hear is typical of what the person's clips will be. Also notes when each person is first heard."""
    import soundfile as sf
    outdir = CFG.runs / name / "speakers"; outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("*.wav"): old.unlink()
    for p in people:
        turns = sorted((fkey, a, b) for fkey, labels in p["members"] for a, b, lab in per_file[fkey]["segments"] if lab in labels)
        p["first"] = round(turns[0][1], 1) if turns else None
        cands = [(fk, a, min(b, a + MAX_SAMPLE_S)) for fk, a, b in turns if b - a >= MIN_SAMPLE_S] or [(fk, a, min(b, a + MAX_SAMPLE_S)) for fk, a, b in turns if b - a >= 0.8]
        pick = [cands[round(i * (len(cands) - 1) / max(1, N_SAMPLES - 1))] for i in range(min(N_SAMPLES, len(cands)))] if cands else []
        p["samples"], p["sample_times"], p["sample_ranges"] = [], [], []
        for fkey, a, b in dict.fromkeys(pick):
            y = _grab(sep, Path(per_file[fkey]["wav"]), a, b)
            fn = f"person{p['id']}_{len(p['samples']) + 1}.wav"; sf.write(outdir / fn, y, SR, subtype="PCM_16")
            p["samples"].append(fn); p["sample_times"].append(round(a, 1)); p["sample_ranges"].append([fkey, round(a, 2), round(b, 2)])


SCAN_HOURS = 40            # how much recording audio the scan listens to by default: an even sample is plenty to find a speaker (0 = everything)


def cache_wav(name, key): return CFG.runs / name / "diar" / (re.sub(r"[^A-Za-z0-9._-]", "_", key) + ".16k.wav")


def scan_pick(files, hours, log=print, durs=None):
    """An evenly spread subset of the recordings that adds up to about `hours` hours (all of them when they fit, or hours is 0)."""
    if not hours or len(files) < 2: return list(files), []
    durs = [(durs or {}).get(f) or audio.duration(f) / 3600 for f in files]; total = sum(durs)
    if total <= hours: return list(files), []
    frac, seen, kept, pick = hours / total, 0.0, 0.0, []
    for f, d in zip(files, durs):
        seen += d
        if kept + d <= frac * seen + d * 0.5: kept += d; pick.append(f)
    pick = pick or [files[0]]
    log(f"There are {total:.3g} hours of recordings. Scanning an even sample of {len(pick)} of {len(files)} (about {sum(d for f, d in zip(files, durs) if f in set(pick)):.3g} hours): "
        "that is plenty to find your speaker. The recordings that are not scanned are not used.")
    return pick, [f for f in files if f not in set(pick)]


def run(name, num_speakers=None, log=print, pipeline=None):
    files = audio_files(name)
    if not files: raise SystemExit(f"No audio files found in {voices.source_dir(name) or 'a folder (choose your recordings folder first)'}")
    every = list(files); hrs = voices.read(name).get("scan_hours"); hrs = SCAN_HOURS if hrs is None else hrs
    from . import analyze
    files, left_out = scan_pick(files, hrs, log, durs=analyze.durations(name, files))
    pipe = pipeline or _load_pipeline()
    cache = CFG.runs / name / "diar"; per_file = {}; fps = []; skipped = []     # fps: (file key, label, embedding)
    prog = Progress(name, "speakers")
    # pyannote reports (step, completed, total) while it works: turn that into smooth progress for the whole job, even for one long file
    WEIGHTS = {"segmentation": (0.0, 0.4, "Finding where people speak"), "speaker_counting": (0.4, 0.05, "Counting the speakers"), "embeddings": (0.45, 0.5, "Recognising each voice")}
    cur = {"i": 1, "t": 0.0}
    def hook(step_name, step_artifact, file=None, total=None, completed=None):
        base, w, label = WEIGHTS.get(step_name, (0.95, 0.05, "Finishing this recording"))
        part = (completed / total) if (completed is not None and total) else 1.0
        if time.time() - cur["t"] < 0.4 and part < 1.0: return           # do not write the file on every batch
        cur["t"] = time.time(); i = cur["i"]
        prog.update(label, i - 1, len(files), files[i - 1].name, frac=((i - 1) + min(1.0, base + w * part)) / len(files))
    for i, f in enumerate(files, 1):
        cur["i"] = i
        key = rel(name, f); wav = cache_wav(name, key)
        prog.update("Finding who speaks in each recording", i - 1, len(files), key)
        try:
            if not (wav.exists() and wav.stat().st_mtime >= f.stat().st_mtime): audio.decode(f, wav, SR, log=log)
            kw = {"num_speakers": int(num_speakers)} if num_speakers else {}
            out = pipe(str(wav), hook=hook, **kw)
            labels = list(out.speaker_diarization.labels())
            emb = out.speaker_embeddings
            segs = [[round(t.start, 3), round(t.end, 3), lab] for t, _, lab in out.exclusive_speaker_diarization.itertracks(yield_label=True)]
        except (audio.AudioError, RuntimeError, ValueError, OSError) as e:
            skipped.append((key, str(e))); log(f"[{i}/{len(files)}] SKIPPED {key}: {e}"); continue
        per_file[key] = {"wav": str(wav), "segments": segs}
        for k, lab in enumerate(labels):
            if emb is not None and k < len(emb) and any(s[2] == lab for s in segs): fps.append((key, lab, emb[k]))
        log(f"[{i}/{len(files)}] {key}: {len(labels)} speaker(s)")
    if not fps: raise SystemExit("No speech found." + (f" {len(skipped)} file(s) could not be read." if skipped else ""))
    ids = cluster([e for _, _, e in fps], float(CFG["speaker_match_distance"]), [k for k, _, _ in fps])
    # largest group first = Person 1
    secs = {}
    for (key, lab, _), c in zip(fps, ids):
        secs[c] = secs.get(c, 0.0) + sum(b - a for a, b, l in per_file[key]["segments"] if l == lab)
    order = sorted(secs, key=lambda c: -secs[c]); num = {c: i + 1 for i, c in enumerate(order)}
    people = {}
    for (key, lab, _), c in zip(fps, ids):
        p = people.setdefault(num[c], {"id": num[c], "seconds": 0.0, "members": {}})
        p["seconds"] += sum(b - a for a, b, l in per_file[key]["segments"] if l == lab)
        p["members"].setdefault(key, []).append(lab)
    plist = []
    for pid in sorted(people):
        p = people[pid]; p["members"] = [[k, v] for k, v in p["members"].items()]; plist.append(p)
    log("Making the samples (music and background noise removed, so you hear just the voices)...")
    sep = _separator(log)
    try: _samples(name, plist, per_file, log, sep)
    finally:
        if sep: sep.close()
    for p in plist:
        for fkey, labels in p["members"]:
            for s in per_file[fkey]["segments"]:
                if s[2] in labels: s[2] = f"P{p['id']}"                 # replace per-file labels by person ids
    out = {"people": [{"id": p["id"], "minutes": round(p["seconds"] / 60, 1), "files": len(p["members"]), "samples": p["samples"], "sample_times": p["sample_times"], "sample_ranges": p["sample_ranges"], "first": p["first"]} for p in plist],
           "files": {k: {"segments": v["segments"]} for k, v in per_file.items()},
           "scan": {"scanned": len(per_file), "total": len(every), "skipped": [rel(name, f) for f in left_out]}}
    result_path(name).write_text(json.dumps(out))
    log("Measuring each voice's pitch...")
    try: person_pitches(name)
    except Exception as e: log(f"(pitch not measured: {e})")
    m = voices.read(name); keep = [i for i in m.get("keep_people", []) if i <= len(plist)]
    voices.write(name, keep_people=keep, use_match=False, people_names={})                 # a new scan starts fresh: any earlier "find more like these" result is gone
    log(f"\nDone: {len(plist)} different voice(s) found. Listen to each and choose who to train on.")
    if skipped:
        log(f"\n{len(skipped)} file(s) were skipped:"); [log(f"  - {k}: {m}") for k, m in skipped[:25]]
    prog.finish(f"{len(plist)} voices found")
    return out


# speaking pitch (Hz): men ~85-155, women ~165-255. Wide overlap zones so only clearly wrong voices are ruled out
FIT = {"male": (0, 170), "female": (145, 999)}               # a whole person (median pitch over many turns)
FIT_SEG = {"male": (0, 178), "female": (138, 999)}           # a single stretch (noisier, so more forgiving)


def pitch_hz(y):
    """Median speaking pitch of mono 16 kHz audio in Hz (or None when there is too little voiced speech). Fast."""
    import numpy as np, librosa
    if len(y) < 4000: return None
    f = librosa.yin(y, fmin=65, fmax=400, sr=SR, frame_length=1024)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0][:len(f)]
    f = f[:len(rms)][rms > np.median(rms)]                  # louder half of the frames = the voiced speech
    return float(np.median(f)) if len(f) >= 5 else None


def fits(voice_type, hz, seg=False):
    if voice_type not in FIT or hz is None: return True
    lo, hi = (FIT_SEG if seg else FIT)[voice_type]
    return lo <= hz <= hi


def person_pitches(name, force=False):
    """Fill in each person's typical pitch (from their longest turns). Quick: runs on the saved scan, no GPU."""
    import numpy as np, soundfile as sf
    r = load(name)
    if not r: return None
    key0 = next(iter(r["files"]), None)
    for p in r["people"]:
        if "sample_ranges" not in p and key0 and len(r["files"]) == 1 and p.get("sample_times"):      # a scan from before clips could be ticked: rebuild where each sample came from
            outdir = CFG.runs / name / "speakers"
            try: p["sample_ranges"] = [[key0, a, round(a + (sf.info(outdir / f).duration), 2)] for a, f in zip(p["sample_times"], p["samples"])]
            except (OSError, RuntimeError): pass
        if p.get("pitch") and not force: continue
        pid, ys = f"P{p['id']}", []
        turns = sorted(((b - a, key, a) for key, v in r["files"].items() for a, b, l in v["segments"] if l == pid and b - a >= 1.0), reverse=True)[:14]
        for d, key, a in turns:
            w = _wavpath(name, key)
            if w.exists(): ys.append(sf.read(w, start=int(a * SR), stop=int((a + min(d, 4)) * SR), dtype="float32")[0])
        hz = pitch_hz(np.concatenate(ys)) if ys else None
        p["pitch"] = round(hz) if hz else None
    result_path(name).write_text(json.dumps(r))
    return r


MATCH_MIN_S, MATCH_MAX_S = 1.0, 15.0
CUSHION = 1.5      # "find more like these" keeps up to this many times your minutes limit, so weak clips can be dropped later
STRICT = {"strict": 0.42, "normal": 0.50, "loose": 0.58}               # how close (cosine distance) a stretch must be to your marked clips


def _wavpath(name, key): return CFG.runs / name / "diar" / (re.sub(r"[^A-Za-z0-9._-]", "_", key) + ".16k.wav")


def _embedder(pipe):
    """Voice fingerprint of a stretch of one recording (the same model the speaker scan uses)."""
    import numpy as np, soundfile as sf, torch
    em = pipe._embedding
    def f(wav, a, b):
        y, _ = sf.read(wav, start=int(a * SR), stop=int(b * SR), dtype="float32")
        if len(y) < SR // 2: return None
        v = np.asarray(em(torch.from_numpy(y)[None, None].to(em.device)), dtype="float64")[0]
        return None if not np.isfinite(v).all() else v / (np.linalg.norm(v) + 1e-9)
    return f


def match(name, log=print, pipeline=None):
    """Find more speech that sounds like the clips you ticked (and not like the ones you crossed), across every person in the recording."""
    import numpy as np, soundfile as sf
    r = load(name)
    if not r: raise SystemExit("Find the speakers first.")
    m = voices.read(name); yes, no = m.get("ref_yes") or [], m.get("ref_no") or []
    if not yes: raise SystemExit("Tick at least one clip (this is my speaker) first.")
    thr = STRICT.get(m.get("strict"), STRICT["normal"])
    pipe = pipeline or _load_pipeline(); embed = _embedder(pipe); prog = Progress(name, "match")
    def ref(items):
        out = []
        for key, a, b in items:
            w = _wavpath(name, key)
            e = embed(w, a, b) if w.exists() else None
            if e is not None: out.append(e)
        return out
    P, N = ref(yes), ref(no)
    if not P: raise SystemExit("The ticked clips could not be read. Find the speakers again, then tick clips.")
    c = np.mean(P, axis=0); c /= np.linalg.norm(c) + 1e-9
    log(f"Learning your voice from {len(P)} ticked clip(s)" + (f", avoiding {len(N)} crossed clip(s)." if N else "."))
    cands = [(key, a, b) for key, v in r["files"].items() for a, b, l in v["segments"] if b - a >= MATCH_MIN_S]
    vt = m.get("voice_type") or ""; acc, t0, wrong = [], 0.0, 0
    if vt: log(f"Looking for a {vt} voice: stretches clearly outside that pitch range are skipped.")
    for i, (key, a, b) in enumerate(cands):
        if time.time() - t0 > 0.5: t0 = time.time(); prog.update("Comparing every voice with your ticked clips", i, len(cands), frac=i / max(1, len(cands)))
        if vt:
            import soundfile as sf2
            hz = pitch_hz(sf2.read(_wavpath(name, key), start=int(a * SR), stop=int(min(b, a + 8) * SR), dtype="float32")[0])
            if not fits(vt, hz, seg=True): wrong += 1; continue
        e = embed(_wavpath(name, key), a, min(b, a + MATCH_MAX_S))
        if e is None: continue
        d = 1 - float(e @ c)
        if N and min(1 - float(e @ n) for n in N) < d: continue          # closer to something you crossed than to your voice
        if d <= thr: acc.append((key, a, b, d))
    if wrong: log(f"Skipped {wrong} stretch(es) that were clearly not a {vt} voice.")
    found_s = sum(b - a for _, a, b, _ in acc)
    cap = int(m.get("max_minutes") or 180); keep_s = cap * CUSHION * 60            # keep the closest stretches only: later steps then work on far less audio
    if found_s > keep_s:
        kept, tot = [], 0.0
        for x in sorted(acc, key=lambda x: x[3]):
            if tot + (x[2] - x[1]) > keep_s: continue
            kept.append(x); tot += x[2] - x[1]
        log(f"Found {found_s / 60:.0f} min that sound like your speaker. Keeping the closest {tot / 60:.0f} min (your {cap}-minute limit plus some room to drop weak clips later).")
        acc = kept
    if not acc:
        raise SystemExit(f"Nothing sounded close enough to your ticked clips at '{m.get('strict') or 'normal'}'. Tick more (and clearer) clips, or try Loose.")
    files = {}
    for key, a, b, _ in sorted(acc):
        lst = files.setdefault(key, [])
        if lst and a - lst[-1][1] <= 0.5: lst[-1][1] = max(lst[-1][1], b)
        else: lst.append([round(a, 2), round(b, 2)])
    minutes = sum(b - a for v in files.values() for a, b in v) / 60; stretches = sum(len(v) for v in files.values())
    # samples to check next: spread through the matches, plus the least sure ones (those are the most useful to tick or cross)
    byt = sorted(x for x in acc if x[2] - x[1] >= MIN_SAMPLE_S) or sorted(acc)
    spread = [byt[round(i * (len(byt) - 1) / max(1, 5))] for i in range(min(6, len(byt)))]
    edge = [x for x in sorted(acc, key=lambda x: -x[3]) if x[2] - x[1] >= MIN_SAMPLE_S and x not in spread][:3]
    outdir = CFG.runs / name / "speakers"; outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("match_*.wav"): old.unlink()
    samples = []; sep = _separator(log)
    for border, group in ((False, spread), (True, edge)):
        for key, a, b, d in dict.fromkeys(group):
            b = min(b, a + MAX_SAMPLE_S); y = _grab(sep, _wavpath(name, key), a, b)
            fn = f"match_{len(samples) + 1}.wav"; sf.write(outdir / fn, y, SR, subtype="PCM_16")
            samples.append({"file": fn, "time": round(a, 1), "range": [key, round(a, 2), round(b, 2)], "border": border})
    if sep: sep.close()
    prev = r.get("match"); history = list((prev or {}).get("history") or [])
    def covered(a, b):                                                   # seconds of audio that both results contain
        return sum(max(0.0, min(x1, y1) - max(x0, y0)) for k, iv in a.items() for x0, x1 in iv for y0, y1 in b.get(k, []))
    secs = lambda f: sum(b - a for v in f.values() for a, b in v)
    settled = False
    if prev:
        both = covered(files, prev["files"]); union = secs(files) + secs(prev["files"]) - both
        settled = union > 0 and both / union >= 0.95                      # essentially the same audio as last round
    history.append(round(minutes, 1))
    r["match"] = {"files": files, "minutes": round(minutes, 1), "stretches": stretches, "strict": m.get("strict") or "normal", "found": round(found_s / 60, 1), "cap": cap, "samples": samples, "settled": settled, "history": history}
    result_path(name).write_text(json.dumps(r))
    voices.write(name, use_match=True)
    log(f"\nDone: found {minutes:.1f} min of speech ({stretches} stretches) that sounds like your ticked clips.")
    if settled: log("Nothing new was found: this is almost exactly the same as the last round. Your voice has settled, and running it again will not change anything.")
    log("Listen to the new samples (the ones marked '?' are the least sure). Tick or cross some more and press the button again to improve it.")
    prog.finish(f"{minutes:.1f} min found")
    return r["match"]


def score_path(name): return CFG.runs / name / "clipscores.json"


def score_clips(name, log=print, pipeline=None, frac_range=(0.0, 1.0), prog=None):
    """How much each prepared clip sounds like the others (voice fingerprint distance from the typical clip). Clips from a different
    person, or with a lot of noise, stand out. Needs the same speaker model as "Who is speaking?"."""
    import numpy as np, soundfile as sf, torch, librosa
    root = CFG.data / name; md = root / "metadata.csv"
    if not md.exists(): raise SystemExit("Prepare the recordings first.")
    files = [l.split("|", 1)[0] for l in md.read_text(encoding="utf-8").splitlines() if "|" in l]
    if len(files) < 5: raise SystemExit("Need at least 5 clips to compare them.")
    pipe = pipeline or _load_pipeline(); em = pipe._embedding; prog = prog or Progress(name, "score"); vec, ok = [], []
    t0 = 0.0
    for i, f in enumerate(files):
        if time.time() - t0 > 0.5: t0 = time.time(); prog.update("Listening to each clip", i, len(files), frac=frac_range[0] + (frac_range[1] - frac_range[0]) * i / len(files))
        try:
            y, sr = sf.read(root / "wavs" / f, dtype="float32"); y = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
            v = np.asarray(em(torch.from_numpy(y)[None, None].to(em.device)), dtype="float64")[0]
        except (OSError, RuntimeError, ValueError): continue
        if np.isfinite(v).all(): vec.append(v / (np.linalg.norm(v) + 1e-9)); ok.append(f)
    if len(vec) < 5: raise SystemExit("Could not read enough clips.")
    x = np.asarray(vec); c = x.mean(0); c /= np.linalg.norm(c) + 1e-9
    for _ in range(2):                                                    # a robust centre: leave out the quarter furthest away, then recompute
        d = 1 - x @ c; keep_ = d <= np.percentile(d, 75); c = x[keep_].mean(0); c /= np.linalg.norm(c) + 1e-9
    d = 1 - x @ c; med = float(np.median(d)); mad = float(np.median(np.abs(d - med))) * 1.4826 + 1e-6
    out = {}
    for f, di in zip(ok, d):
        z = (float(di) - med) / mad
        out[f] = {"d": round(float(di), 3), "lv": 2 if (z > 3 and di > med + 0.08) else 1 if (z > 2 and di > med + 0.05) else 0}
    score_path(name).write_text(json.dumps({"scores": out, "median": round(med, 3)}))
    bad = sum(1 for v in out.values() if v["lv"] == 2); maybe = sum(1 for v in out.values() if v["lv"] == 1)
    log(f"Done: {len(out)} clips checked. {bad} sound very different from the rest, {maybe} a bit different.")
    if frac_range == (0.0, 1.0): prog.finish(f"{bad} flagged")
    return out


def keep(name, ids):
    r = load(name)
    if not r: raise SystemExit("Find the speakers first.")
    valid = {p["id"] for p in r["people"]}
    bad = [i for i in ids if i not in valid]
    if bad: raise SystemExit(f"No such person: {bad}")
    voices.write(name, keep_people=sorted(set(ids)))


def stale(name):
    """True if the recordings changed since the speakers were last found."""
    r = load(name)
    return bool(r) and set(r["files"]) | set((r.get("scan") or {}).get("skipped", [])) != {rel(name, f) for f in audio_files(name)}


def intervals(name, key, gap=0.5):
    """(start, end) seconds where the chosen people speak in one file, or None when no one was chosen."""
    m = voices.read(name); ids = m.get("keep_people") or []
    r = load(name)
    if m.get("use_match") and r and r.get("match"):                     # "find more like these": the stretches that sounded like the marked clips
        return [(a, b) for a, b in sorted(r["match"]["files"].get(key, []))]
    if not ids or not r: return None
    if key not in r["files"]: raise SystemExit(f"'{key}' was added after the speakers were found: press 'Find the speakers' again.")
    want = {f"P{i}" for i in ids}
    iv = sorted((a, b) for a, b, l in r["files"][key]["segments"] if l in want)
    out = []
    for a, b in iv:
        if out and a - out[-1][1] <= gap: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return [(a, b) for a, b in out]
