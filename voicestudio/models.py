"""A copy of the gated speaker model, kept on this computer and kept up to date.

The copy lives in the data folder (models/), downloaded once with your own Hugging Face access. It is never part of this program's
repository: each person accepts the model's terms themselves. The copy lets the speaker steps start without asking Hugging Face
every time and keeps them working offline. Roughly once a day (and whenever you press Update) the program asks Hugging Face
whether a newer version exists and, if so, installs it; if Hugging Face cannot be reached it simply keeps using the copy it has.
"""
import json, shutil, time
from .config import CFG

CHECK_EVERY = 24 * 3600          # seconds between update checks


def dir_for(model): return CFG.home / "models" / model.replace("/", "__")


def _meta(d):
    try: return json.loads((d / ".piper-mill.json").read_text())
    except (OSError, ValueError): return {}


def _save(d, sha): (d / ".piper-mill.json").write_text(json.dumps({"sha": sha, "checked": time.time()}))


def status(model):
    """What the page shows: whether a copy exists, which revision it is, and when it was last checked."""
    d = dir_for(model); m = _meta(d)
    return {"copy": (d / "config.yaml").exists(), "revision": (m.get("sha") or "")[:7], "checked": m.get("checked")}


def ensure(model, token, log=print, force=False):
    """Path to the config.yaml of a local copy of `model`: downloaded the first time, updated when a newer version exists."""
    import huggingface_hub as H
    d = dir_for(model); assert d.parent == CFG.home / "models"
    have = (d / "config.yaml").exists(); m = _meta(d)
    if have and not force and time.time() - m.get("checked", 0) < CHECK_EVERY: return d / "config.yaml"
    try:
        sha = H.HfApi().model_info(model, token=token).sha
    except Exception:
        if have: log("Could not check for a newer speaker model (no connection?). Using the copy stored on this computer."); return d / "config.yaml"
        raise
    if have and m.get("sha") == sha: _save(d, sha); return d / "config.yaml"
    log("Downloading the speaker model (about 40 MB, once)..." if not have else "A newer speaker model is available. Updating the stored copy...")
    tmp = d.with_name(d.name + ".new"); shutil.rmtree(tmp, ignore_errors=True)
    H.snapshot_download(model, local_dir=tmp, token=token, ignore_patterns=["*.gif", "*.md", ".gitattributes"])
    if not (tmp / "config.yaml").exists(): shutil.rmtree(tmp, ignore_errors=True); raise RuntimeError("The download was incomplete.")
    _save(tmp, sha)
    if have:
        old = d.with_name(d.name + ".old"); shutil.rmtree(old, ignore_errors=True); d.rename(old)
    d.parent.mkdir(parents=True, exist_ok=True); tmp.rename(d)
    if have: shutil.rmtree(d.with_name(d.name + ".old"), ignore_errors=True)
    return d / "config.yaml"


def newer_available(model, token):
    """True when Hugging Face has a newer version than the stored copy (False when it cannot tell)."""
    import huggingface_hub as H
    try: return bool(status(model)["copy"]) and H.HfApi().model_info(model, token=token).sha != _meta(dir_for(model)).get("sha")
    except Exception: return False
