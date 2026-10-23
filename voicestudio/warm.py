"""Keep the models this program needs on this computer, and up to date.

When the web page starts, a background thread downloads anything missing and asks Hugging Face whether a newer version of
each model exists. The speaker model is checked again every time it is used (models.ensure); the speech-to-text model is
checked by transformers each time it is loaded. With no connection the stored copies are simply used. The page lists what is stored
under Voice options -> Stored models, where a copy can be deleted (it downloads again when it is next needed or on the next start)."""
import shutil, threading
from pathlib import Path
from .config import CFG
from . import base, models

STATE = {}            # id -> short text for the page ("checking", "up to date", "downloading", "offline: using stored copy", ...)
_lock = threading.Lock()


def _set(k, v):
    with _lock: STATE[k] = v


def _hf_cache(model): return Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + model.replace("/", "--"))


def _size(p):
    p = Path(p)
    if p.is_file(): return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def whisper(log=print):
    import huggingface_hub as H
    m = CFG["whisper_model"]
    H.snapshot_download(m, allow_patterns=["*.json", "*.txt", "model.safetensors"])


def listing():
    """What is stored, for the page: [{id, name, size, note}]."""
    out = []
    sm = CFG["diarization_model"]; st = models.status(sm)
    out.append({"id": "pyannote", "name": "Speaker model", "size": _size(models.dir_for(sm)), "present": st["copy"], "note": ("revision " + st["revision"]) if st["revision"] else ""})
    wm = CFG["whisper_model"]
    out.append({"id": "whisper", "name": "Speech-to-text model (Whisper)", "size": _size(_hf_cache(wm)), "present": _hf_cache(wm).exists(), "note": ""})
    for key, (label, _) in base.BASES.items():
        p = base.path(key)
        out.append({"id": "base:" + key, "name": "Starting voice: " + label, "size": _size(p), "present": p.exists(), "note": ""})
    for o in out: o["state"] = STATE.get(o["id"], "")
    return out


def delete_all():
    """Delete every stored model. They download again at the next start."""
    for m in listing():
        if m["present"]: delete(m["id"])


def delete(mid):
    if mid == "pyannote": shutil.rmtree(models.dir_for(CFG["diarization_model"]), ignore_errors=True)
    elif mid == "whisper": shutil.rmtree(_hf_cache(CFG["whisper_model"]), ignore_errors=True)
    elif mid.startswith("base:") and mid[5:] in base.BASES:
        p = base.path(mid[5:]); p.unlink(missing_ok=True); p.with_suffix(".json").unlink(missing_ok=True)
    else: raise ValueError("unknown model")
    STATE.pop(mid, None)


def run():
    """Start-up check: download what is missing, update what is old. Never raises; the page keeps working offline."""
    from .speakers import _token
    tok = _token()
    if tok:
        _set("pyannote", "checking")
        try:
            models.ensure(CFG["diarization_model"], tok, log=lambda t: None, force=True); _set("pyannote", "up to date")
        except Exception: _set("pyannote", "could not check (offline or no access): using the stored copy" if models.status(CFG["diarization_model"])["copy"] else "not downloaded: needs access")
    _set("whisper", "checking")
    try: whisper(); _set("whisper", "up to date")
    except Exception: _set("whisper", "could not check (offline): using the stored copy" if _hf_cache(CFG["whisper_model"]).exists() else "not downloaded")
    for key in base.BASES:
        mid = "base:" + key
        if base.path(key).exists() and base.info(key): _set(mid, "ready"); continue
        _set(mid, "downloading")
        try: base.ensure(key, log=lambda t: None); _set(mid, "ready")
        except Exception: _set(mid, "not downloaded (offline?)")


def start(): threading.Thread(target=run, daemon=True, name="warm").start()
