"""Newest checkpoint -> Piper voice files (name.onnx + name.onnx.json) and a spoken test clip."""
import json, shutil, subprocess, sys
from .config import CFG
from .progress import Progress
from .train import pick

DEFAULT_TEXT = "Good evening. This is a test of my new voice, and I hope it sounds like me."


def _label_json(path, name):
    """Piper/Home Assistant list a voice by its json: add the language block and a friendly dataset name (a bare training config has neither, so HA could not offer the voice)."""
    j = json.loads(path.read_text())
    lang = j.setdefault("language", {})
    lang.setdefault("code", "en_US"); lang.setdefault("family", "en"); lang.setdefault("region", "US")
    lang.setdefault("name_native", "English"); lang.setdefault("name_english", "English"); lang.setdefault("country_english", "United States")
    j.setdefault("dataset", name)
    path.write_text(json.dumps(j, indent=1))


def run(name, text=DEFAULT_TEXT, which="best"):
    ck, what = pick(name, which)
    if not ck: raise SystemExit("Training has not saved a checkpoint yet. Train the voice first.")
    prog = Progress(name, "export"); prog.update("Converting the voice", 0, 2)
    CFG.out.mkdir(parents=True, exist_ok=True)
    onnx = CFG.out / f"{name}.onnx"
    print(f"Exporting the {what} ({ck.name}) ...", flush=True)
    subprocess.run([sys.executable, "-m", "piper.train.export_onnx", "--checkpoint", str(ck), "--output-file", str(onnx)], check=True)
    shutil.copy(CFG.runs / name / "config.json", CFG.out / f"{name}.onnx.json")
    _label_json(CFG.out / f"{name}.onnx.json", name)
    (CFG.out / f"{name}.version.txt").write_text(f"{what} ({ck.name})")        # so the page can say which version you are listening to
    prog.update("Speaking the test sentence", 1, 2)
    clip = CFG.out / f"{name}-test.wav"
    subprocess.run([sys.executable, "-m", "piper", "-m", str(onnx), "-f", str(clip), "--", text or DEFAULT_TEXT], check=True)
    prog.finish("done")
    print(f"Done: {onnx} (+ .json) and test clip {clip.name}", flush=True)
