"""A 'voice' is a folder set under the data home plus a small JSON file (base voice, optional source folder)."""
import json, shutil
from pathlib import Path
from .config import CFG


def meta_path(n): return CFG.voices / f"{n}.json"


def exists(n): return meta_path(n).exists()


def read(n):
    try: return json.loads(meta_path(n).read_text())
    except (OSError, ValueError): return {}


def write(n, **kw):
    CFG.ensure_dirs()
    m = read(n); m.update(kw); meta_path(n).write_text(json.dumps(m, indent=1))
    return m


def names(): return sorted(p.stem for p in CFG.voices.glob("*.json")) if CFG.voices.exists() else []


def source_dir(n):
    """The folder (on this computer) the recordings are read from, or None if it has not been chosen yet."""
    s = read(n).get("source_dir")
    return Path(s).expanduser() if s else None


def delete(n):
    """Remove everything this tool created for a voice. Your recordings folder is never touched."""
    for p in (CFG.data / n, CFG.runs / n, CFG.cache / n):
        shutil.rmtree(p, ignore_errors=True)
    for f in (CFG.out.glob(f"{n}.*"), CFG.out.glob(f"{n}-*")):
        for p in f: p.unlink(missing_ok=True)
    meta_path(n).unlink(missing_ok=True)
