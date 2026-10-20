"""Back up a whole voice to one zip, and restore it (under any name).

A backup holds: the voice's settings, its prepared clips and transcript, its speaker/clip scores and training settings, the exported voice
files, and (optionally) the training checkpoints, which are large. It leaves out the decoded-audio cache, which can be rebuilt. Your original
recordings are never inside a backup.
"""
import json, os, re, shutil, time, zipfile
from pathlib import Path
from . import voices
from .config import CFG
from .progress import Progress

FORMAT = "piper-mill-backup"


class Sink:
    """A write-only stream for zipfile: the zip is sent to the browser as it is made, so even a many-gigabyte backup needs no temporary file."""
    def __init__(self, raw): self.raw, self.n = raw, 0
    def write(self, b): self.raw.write(b); self.n += len(b); return len(b)
    def tell(self): return self.n
    def flush(self): self.raw.flush()


def files_for(name, checkpoints):
    out = [("voice.json", voices.meta_path(name))]
    for base, top in ((CFG.data, "data"),):
        for p in sorted((base / name).rglob("*")):
            if p.is_file(): out.append((f"{top}/{name}/{p.relative_to(base / name).as_posix()}", p))
    runs = CFG.runs / name
    if runs.exists():
        for p in sorted(runs.rglob("*")):
            rel = p.relative_to(runs)
            if not p.is_file() or rel.parts[0] in ("diar", "clean") or p.suffix in (".log", ".pid", ".tmp") or p.name.endswith(".progress.json"): continue
            if rel.parts[0] == "lightning_logs" and not checkpoints: continue
            out.append((f"runs/{name}/{rel.as_posix()}", p))
    for p in sorted(CFG.out.glob(f"{name}.*")) + sorted(CFG.out.glob(f"{name}-compare*")): out.append((f"out/{p.name}", p))
    return [(a, p) for a, p in out if p.exists()]


def write_zip(name, checkpoints, raw):
    items = files_for(name, checkpoints)
    with zipfile.ZipFile(Sink(raw), "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        z.writestr("piper-mill-backup.json", json.dumps({"format": FORMAT, "version": 1, "name": name, "made": time.strftime("%Y-%m-%d %H:%M"), "checkpoints": bool(checkpoints), "files": len(items)}))
        for arc, p in items: z.write(p, arc)


def check(zpath):
    """Open a backup zip and return (zipfile, manifest), or say plainly what is wrong with it."""
    try: zf = zipfile.ZipFile(Path(zpath).expanduser())
    except (OSError, zipfile.BadZipFile): raise SystemExit("That is not a readable zip file.")
    try: man = json.loads(zf.read("piper-mill-backup.json"))
    except (KeyError, ValueError): raise SystemExit("That zip is not a Piper Mill backup (it has no piper-mill-backup.json).")
    if man.get("format") != FORMAT: raise SystemExit("That zip is not a Piper Mill backup.")
    return zf, man


def restore(name, zpath, log=print):
    zf, man = check(zpath)
    old = man["name"]; members = [m for m in zf.infolist() if m.filename != "piper-mill-backup.json"]; total = sum(m.file_size for m in members) or 1
    prog = Progress(name, "restore"); done = 0
    for m in members:
        parts = m.filename.split("/")
        if m.is_dir() or ".." in parts or m.filename.startswith("/"): continue
        if parts[0] == "voice.json": dest = voices.meta_path(name)
        elif parts[0] in ("data", "runs") and len(parts) >= 3 and parts[1] == old:
            leaf = parts[-1]
            if leaf.startswith(old + "_"): leaf = name + leaf[len(old):]
            dest = (CFG.data if parts[0] == "data" else CFG.runs) / name / Path(*parts[2:-1]) / leaf
        elif parts[0] == "out" and len(parts) == 2 and (parts[1].startswith(old + ".") or parts[1].startswith(old + "-")): dest = CFG.out / (name + parts[1][len(old):])
        else: continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(m) as s, open(dest, "wb") as d: shutil.copyfileobj(s, d, 1 << 20)
        done += m.file_size; prog.update("Restoring the files", done, total, m.filename[:60], frac=done / total)
    root = CFG.data / name
    for fn in ("metadata.csv", "metadata.orig.csv"):
        f = root / fn
        if f.exists():
            f.write_text("\n".join((name + l[len(old):] if l.startswith(old + "_") else l) for l in f.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    shutil.rmtree(CFG.cache / name, ignore_errors=True)
    m = voices.read(name); m.pop("restoring", None); m.pop("words_seen", None)
    voices.meta_path(name).write_text(json.dumps(m, indent=1))
    log(f"Done: restored {len(members)} files as '{name}'" + ("" if man.get("checkpoints") else " (this backup has no training checkpoints, so training starts again from the starting voice)") + ".")
    prog.finish("restored")
