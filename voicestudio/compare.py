"""Make the same test sentence with each version of a trained voice (best sounding, most accurate, latest) so you can listen and choose.

Each version is exported to a temporary file and spoken; your real exported voice (name.onnx) is not touched. When two versions are the
same checkpoint it is only made once.
"""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path
from .config import CFG
from .export import DEFAULT_TEXT
from .progress import Progress
from .train import pick

VERSIONS = [("best", "Best sounding"), ("accurate", "Most accurate"), ("latest", "Latest")]


def meta_path(name): return CFG.out / f"{name}-compare.json"


def clip_path(name, i): return CFG.out / f"{name}-compare-{i}.wav"


def run(name, text=DEFAULT_TEXT, log=print):
    picks = []
    for w, label in VERSIONS:
        ck, what = pick(name, w)
        if not ck: raise SystemExit("No training checkpoint yet. Train first.")
        picks.append((w, label, ck, what))
    groups = {}
    for w, label, ck, what in picks: groups.setdefault(ck, []).append({"which": w, "label": label, "what": what})
    CFG.out.mkdir(parents=True, exist_ok=True); prog = Progress(name, "compare"); items = []
    for old in CFG.out.glob(f"{name}-compare-*.wav"): old.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for i, (ck, versions) in enumerate(groups.items()):
            prog.update("Making a sample from each version", i, len(groups), ck.name, frac=i / len(groups))
            log(f"[{i + 1}/{len(groups)}] {' / '.join(v['label'] for v in versions)}: {ck.name}")
            onnx = Path(td) / f"v{i}.onnx"
            r = subprocess.run([sys.executable, "-m", "piper.train.export_onnx", "--checkpoint", str(ck), "--output-file", str(onnx)], capture_output=True, text=True)
            if r.returncode: raise SystemExit("Could not prepare one of the versions: " + (r.stderr.strip().splitlines() or ["export failed"])[-1][:160])
            shutil.copy(CFG.runs / name / "config.json", str(onnx) + ".json")
            subprocess.run([sys.executable, "-m", "piper", "-m", str(onnx), "-f", str(clip_path(name, i)), "--", text or DEFAULT_TEXT], check=True, capture_output=True)
            items.append({"checkpoint": ck.name, "versions": versions})
    meta_path(name).write_text(json.dumps({"text": text or DEFAULT_TEXT, "items": items}))
    log(f"\nDone: {len(items)} sample(s) made" + (" (some versions are the same checkpoint)" if len(items) < len(VERSIONS) else "") + ".")
    prog.finish(f"{len(items)} samples")
