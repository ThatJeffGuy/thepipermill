"""Analyze the recordings folder before anything else runs.

Checks that every file can be read, measures how much audio there is, and gets the files ready for the later steps: the recordings
the speaker scan will listen to are converted once, up front, to the format it needs. Your recordings are only read, never changed.
The result is kept in analysis.json next to the voice's other working files and is valid until the folder changes.
"""
import hashlib, json
from . import audio, speakers, voices
from .config import CFG
from .progress import Progress


def path(name): return CFG.runs / name / "analysis.json"


def signature(name, files=None):
    """A fingerprint of the folder's audio files (names, sizes, modified times): the analysis is stale when it changes."""
    h = hashlib.sha256()
    for f in (speakers.audio_files(name) if files is None else files):
        st = f.stat(); h.update(f"{speakers.rel(name, f)}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def load(name):
    try: return json.loads(path(name).read_text())
    except (OSError, ValueError): return None


def durations(name, files):
    """{file: hours} from a current analysis, or {} when there is none (callers then measure the files themselves)."""
    a = load(name)
    if not a or a.get("sig") != signature(name, files): return {}
    return {f: a["files"][speakers.rel(name, f)]["secs"] / 3600 for f in files if a["files"].get(speakers.rel(name, f), {}).get("ok")}


def summary(name, files):
    """What the page needs: totals, files that cannot be read, and whether the folder changed since."""
    a = load(name)
    if not a: return None
    bad = [{"key": k, "err": v.get("err", "")} for k, v in a["files"].items() if not v.get("ok")]
    return {"hours": a["hours"], "readable": a["readable"], "total": a["total"], "unreadable": bad[:20], "sample": a.get("sample", 0),
            "stale": a.get("sig") != signature(name, files)}


def run(name, log=print):
    files = speakers.audio_files(name)
    if not files: raise SystemExit(f"No audio files found in {voices.source_dir(name) or 'a folder (choose your recordings folder first)'}")
    prog = Progress(name, "analyze"); info = {}; n = len(files)
    for i, f in enumerate(files):
        key = speakers.rel(name, f); prog.update("Looking at each recording", i, n, key, frac=0.4 * i / n)
        try:
            audio.probe(f); secs = audio.duration(f)
            if secs <= 0: raise audio.AudioError("has no audible length")
            info[key] = {"ok": True, "secs": round(secs, 1)}
        except audio.AudioError as e:
            info[key] = {"ok": False, "err": str(e)}; log(f"{key}: {e}. It will be left out.")
    good = [f for f in files if info[speakers.rel(name, f)]["ok"]]
    if not good: raise SystemExit("None of the files could be read as audio.")
    hrs = speakers.SCAN_HOURS
    sample, _ = speakers.scan_pick(good, hrs, log, durs={f: info[speakers.rel(name, f)]["secs"] / 3600 for f in good})
    cache = CFG.runs / name / "diar"; cache.mkdir(parents=True, exist_ok=True); ready = 0
    for j, f in enumerate(sample):
        key = speakers.rel(name, f); wav = speakers.cache_wav(name, key)
        prog.update("Getting the recordings ready", j, len(sample), key, frac=0.4 + 0.6 * j / max(1, len(sample)))
        try:
            src = speakers.source(name, f)
            if not (wav.exists() and wav.stat().st_mtime >= src.stat().st_mtime): audio.decode(src, wav, speakers.SR, log=log)
            ready += 1
        except audio.AudioError as e:
            info[key] = {"ok": False, "err": str(e)}; log(f"{key}: {e}. It will be left out.")
    hours = sum(v["secs"] for v in info.values() if v["ok"]) / 3600
    path(name).write_text(json.dumps({"sig": signature(name, files), "files": info, "hours": round(hours, 1), "readable": sum(1 for v in info.values() if v["ok"]),
                                      "total": n, "sample": len(sample)}))
    log(f"\nDone: {sum(1 for v in info.values() if v['ok'])} of {n} recordings can be read, {hours:.3g} hours in all.")
    prog.finish(f"{hours:.0f} hours analyzed")
