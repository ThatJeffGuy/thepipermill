"""Newest checkpoint -> Piper voice files (name.onnx + name.onnx.json) and a spoken test clip."""
import shutil, subprocess, sys
from .config import CFG
from .train import pick

DEFAULT_TEXT = "Good evening. This is a test of my new voice, and I hope it sounds like me."


def run(name, text=DEFAULT_TEXT, which="best"):
    ck, what = pick(name, which)
    if not ck: raise SystemExit("No training checkpoint yet. Train first.")
    CFG.out.mkdir(parents=True, exist_ok=True)
    onnx = CFG.out / f"{name}.onnx"
    print(f"Exporting the {what} ({ck.name}) ...", flush=True)
    subprocess.run([sys.executable, "-m", "piper.train.export_onnx", "--checkpoint", str(ck), "--output-file", str(onnx)], check=True)
    shutil.copy(CFG.runs / name / "config.json", CFG.out / f"{name}.onnx.json")
    (CFG.out / f"{name}.version.txt").write_text(f"{what} ({ck.name})")        # so the page can say which version you are listening to
    clip = CFG.out / f"{name}-test.wav"
    subprocess.run([sys.executable, "-m", "piper", "-m", str(onnx), "-f", str(clip), "--", text or DEFAULT_TEXT], check=True)
    print(f"Done: {onnx} (+ .json) and test clip {clip.name}", flush=True)
