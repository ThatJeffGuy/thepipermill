"""Make the same test sentence with each version of a trained voice (best sounding, most accurate, latest) so you can listen and choose.

Each version is exported to a temporary file and spoken; your real exported voice (name.onnx) is not touched. When two versions are the
same checkpoint it is only made once.
"""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path
from .config import CFG
from .export import DEFAULT_TEXT, _label_json
from .progress import Progress
from .train import pick

VERSIONS = [("best", "Best sounding"), ("accurate", "Most accurate"), ("latest", "Latest")]


# A longer passage than the single test sentence, written to show how a voice handles: low grumbly words, bright excited ones,
# a pause at a comma, a full stop, a question and an exclamation.
COMPARE_TEXT = ("Well, I suppose we should begin. The night is long, the road is longer, and nobody is coming to help us. "
                "Did you really think it would be that easy? Hmm. Fine. Listen closely: low and slow, then suddenly bright and quick! "
                "What a wonderful surprise. Now, take a breath, and tell me, what happens next?")


def meta_path(name): return CFG.out / f"{name}-compare.json"


def clip_path(name, i, speaker=None): return CFG.out / (f"{name}-compare-{i}.wav" if speaker is None else f"{name}-compare-{i}-{speaker}.wav")


def speaker_map(name):
    """{speaker name: id} for a multi-speaker voice (e.g. low / bright), else {} for an ordinary single-speaker voice."""
    try: m = json.loads((CFG.runs / name / "config.json").read_text()).get("speaker_id_map") or {}
    except (OSError, ValueError): return {}
    return m if len(m) > 1 else {}


def run(name, text=COMPARE_TEXT, log=print):
    picks = []
    for w, label in VERSIONS:
        ck, what = pick(name, w)
        if not ck: raise SystemExit("Training has not saved a checkpoint yet. Train the voice first.")
        picks.append((w, label, ck, what))
    groups = {}
    for w, label, ck, what in picks: groups.setdefault(ck, []).append({"which": w, "label": label, "what": what})
    CFG.out.mkdir(parents=True, exist_ok=True); prog = Progress(name, "compare"); items = []
    for old in list(CFG.out.glob(f"{name}-compare-*.wav")) + list(CFG.out.glob(f"{name}-compare-*.onnx*")): old.unlink(missing_ok=True)
    # each version's voice files are kept (name-compare-N.onnx + .json) so the page can offer them as a zip
    for i, (ck, versions) in enumerate(groups.items()):
        prog.update("Making a sample from each version", i, len(groups), ck.name, frac=i / len(groups))
        log(f"[{i + 1}/{len(groups)}] {' / '.join(v['label'] for v in versions)}: {ck.name}")
        onnx = CFG.out / f"{name}-compare-{i}.onnx"
        r = subprocess.run([sys.executable, "-m", "piper.train.export_onnx", "--checkpoint", str(ck), "--output-file", str(onnx)], capture_output=True, text=True)
        if r.returncode: raise SystemExit("Could not prepare one of the versions: " + (r.stderr.strip().splitlines() or ["export failed"])[-1][:160])
        shutil.copy(CFG.runs / name / "config.json", str(onnx) + ".json"); _label_json(Path(str(onnx) + ".json"), name)
        spk = speaker_map(name)
        for sname, sid in (spk.items() or [(None, None)]):       # one sample per speaker for a multi-speaker voice
            out = clip_path(name, i, sname); cmd = [sys.executable, "-m", "piper", "-m", str(onnx), "-f", str(out)] + (["-s", str(sid)] if sname else []) + ["--", text or COMPARE_TEXT]
            subprocess.run(cmd, check=True, capture_output=True)
            items.append({"checkpoint": ck.name, "versions": versions, "file": out.name, "onnx": onnx.name, **({"speaker": sname} if sname else {})})
    meta_path(name).write_text(json.dumps({"text": text or COMPARE_TEXT, "items": items}))
    log(f"\nDone: {len(items)} sample(s) made" + (" (some versions are the same checkpoint)" if len(items) < len(VERSIONS) else "") + ".")
    prog.finish(f"{len(items)} samples")
