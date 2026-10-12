"""Who is speaking?  For recordings with more than one person in them.

1. `run` finds the speakers in every recording (pyannote.audio diarization), then groups the same person across files
   by comparing voice fingerprints, and saves a short sample clip of each person so you can listen and choose.
2. `keep` saves which people to train on.
3. Prepare then uses only the parts where the chosen people speak.

Only use recordings of people who agree to it.

The diarization model (pyannote/speaker-diarization-community-1, CC-BY-4.0) is gated on Hugging Face: accept its
conditions once on its web page and give a read token (config `hf_token`, the HF_TOKEN variable, or `hf auth login`).
"""
import json, os, re, subprocess
from pathlib import Path
from . import audio, voices
from .progress import Progress
from .config import CFG

SR = 16000
EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mkv", ".webm"}
MIN_SAMPLE_S, MAX_SAMPLE_S = 3.0, 8.0


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


def cluster(embeddings, threshold):
    """Group voice fingerprints (one per speaker per file) into people. Returns a list of cluster ids."""
    import numpy as np
    if len(embeddings) == 1: return [0]
    from sklearn.cluster import AgglomerativeClustering
    x = np.asarray(embeddings, dtype="float64"); x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=threshold).fit_predict(x)
    return [int(i) for i in lab]


def _samples(name, people, per_file, log):
    """Write up to 3 short sample clips per person (their longest clean turns)."""
    import soundfile as sf
    outdir = CFG.runs / name / "speakers"; outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("*.wav"): old.unlink()
    for p in people:
        cands = []
        for fkey, labels in p["members"]:
            for a, b, lab in per_file[fkey]["segments"]:
                if lab in labels and b - a >= MIN_SAMPLE_S: cands.append((b - a, fkey, a, min(b, a + MAX_SAMPLE_S)))
        cands.sort(reverse=True); p["samples"] = []
        used = set()
        for _, fkey, a, b in cands:
            if len(p["samples"]) >= 3: break
            if (fkey, int(a // 30)) in used: continue            # spread the samples around
            used.add((fkey, int(a // 30)))
            wav = Path(per_file[fkey]["wav"])
            y, _ = sf.read(wav, start=int(a * SR), stop=int(b * SR), dtype="float32")
            fn = f"person{p['id']}_{len(p['samples']) + 1}.wav"; sf.write(outdir / fn, y, SR, subtype="PCM_16"); p["samples"].append(fn)


def run(name, num_speakers=None, log=print, pipeline=None):
    files = audio_files(name)
    if not files: raise SystemExit(f"No audio files found in {voices.source_dir(name) or 'a folder (choose your recordings folder first)'}")
    pipe = pipeline or _load_pipeline()
    cache = CFG.runs / name / "diar"; per_file = {}; fps = []; skipped = []     # fps: (file key, label, embedding)
    prog = Progress(name, "speakers")
    for i, f in enumerate(files, 1):
        key = rel(name, f); wav = cache / (re.sub(r"[^A-Za-z0-9._-]", "_", key) + ".16k.wav")
        prog.update("Finding who speaks in each recording", i - 1, len(files), key)
        try:
            if not (wav.exists() and wav.stat().st_mtime >= f.stat().st_mtime): audio.decode(f, wav, SR, log=log)
            kw = {"num_speakers": int(num_speakers)} if num_speakers else {}
            out = pipe(str(wav), **kw)
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
    ids = cluster([e for _, _, e in fps], float(CFG["speaker_match_distance"]))
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
    _samples(name, plist, per_file, log)
    for p in plist:
        for fkey, labels in p["members"]:
            for s in per_file[fkey]["segments"]:
                if s[2] in labels: s[2] = f"P{p['id']}"                 # replace per-file labels by person ids
    out = {"people": [{"id": p["id"], "minutes": round(p["seconds"] / 60, 1), "files": len(p["members"]), "samples": p["samples"]} for p in plist],
           "files": {k: {"segments": v["segments"]} for k, v in per_file.items()}}
    result_path(name).write_text(json.dumps(out))
    m = voices.read(name); keep = [i for i in m.get("keep_people", []) if i <= len(plist)]
    voices.write(name, keep_people=keep)
    log(f"\nDone: {len(plist)} different voice(s) found. Listen to each and choose who to train on.")
    if skipped:
        log(f"\n{len(skipped)} file(s) were skipped:"); [log(f"  - {k}: {m}") for k, m in skipped[:25]]
    prog.finish(f"{len(plist)} voices found")
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
    return bool(r) and set(r["files"]) != {rel(name, f) for f in audio_files(name)}


def intervals(name, key, gap=0.5):
    """(start, end) seconds where the chosen people speak in one file, or None when no one was chosen."""
    ids = voices.read(name).get("keep_people") or []
    r = load(name)
    if not ids or not r: return None
    if key not in r["files"]: raise SystemExit(f"'{key}' was added after the speakers were found: press 'Find the speakers' again.")
    want = {f"P{i}" for i in ids}
    iv = sorted((a, b) for a, b, l in r["files"][key]["segments"] if l in want)
    out = []
    for a, b in iv:
        if out and a - out[-1][1] <= gap: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return [(a, b) for a, b in out]
