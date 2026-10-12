"""Command line:  python -m voicestudio <command>   (or just `voicestudio` once installed)"""
import argparse, sys
from . import voices
from .config import CFG


def main(argv=None):
    ap = argparse.ArgumentParser(prog="voicestudio", description="Train your own Piper voice from recordings.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="start the web page")
    sub.add_parser("doctor", help="check the GPU and tools")
    sub.add_parser("list", help="list voices")
    p = sub.add_parser("new", help="create a voice"); p.add_argument("name"); p.add_argument("--base", default="ryan"); p.add_argument("--folder", help="use recordings from this folder (instead of copying files in)")
    p = sub.add_parser("prep", help="recordings -> dataset"); p.add_argument("name"); p.add_argument("--top-db", type=float, default=35.0)
    p = sub.add_parser("train", help="fine-tune (resumable)"); p.add_argument("name"); p.add_argument("epochs", nargs="?", type=int, default=1000); p.add_argument("batch", nargs="?", type=int, default=0)
    p = sub.add_parser("export", help="make the .onnx/.json and a test clip"); p.add_argument("name"); p.add_argument("--text", default=None)
    p = sub.add_parser("install", help="run the configured install step"); p.add_argument("name")
    p = sub.add_parser("delete", help="delete a voice and its working files"); p.add_argument("name")
    a = ap.parse_args(argv)
    CFG.ensure_dirs()

    if a.cmd == "serve":
        from .server import serve; return serve()
    if a.cmd == "doctor":
        from . import gpu; gpu.check()
        import shutil
        for t in ("ffmpeg", "git"): print(f"{t}: {'ok' if shutil.which(t) else 'MISSING'}")
        for mod, what in (("piper.espeakbridge", "espeak-ng bridge"), ("piper.train.vits.monotonic_align", "alignment code"), ("piper.train", "trainer")):
            try: __import__(mod); print(f"{what}: ok")
            except Exception as e: print(f"{what}: PROBLEM ({e}) - run ./install.sh again")
        print("data home:", CFG.home, "| config:", CFG.path if CFG.path.exists() else "(defaults)"); return
    if a.cmd == "list":
        for n in voices.names(): print(n)
        return
    if a.cmd == "new":
        from . import base
        n = CFG.check_name(a.name)
        if voices.exists(n): sys.exit(f"'{n}' already exists")
        base.ensure(a.base)
        voices.write(n, base=a.base, **({"source_dir": str(a.folder)} if a.folder else {}))
        (CFG.inbox / n).mkdir(parents=True, exist_ok=True)
        print(f"Created '{n}'. Recordings go in: {voices.source_dir(n)}"); return
    n = CFG.check_name(a.name)
    if not voices.exists(n): sys.exit(f"No voice '{n}'. Create it first:  voicestudio new {n}")
    if a.cmd == "prep":
        from . import prep; prep.run(n, top_db=a.top_db)
    elif a.cmd == "train":
        from . import train; train.run(n, a.epochs, a.batch)
    elif a.cmd == "export":
        from . import export; export.run(n, a.text or export.DEFAULT_TEXT)
    elif a.cmd == "install":
        from . import install; install.run(n)
    elif a.cmd == "delete":
        voices.delete(n); print(f"Deleted '{n}'.")
