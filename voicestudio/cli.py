"""Command line:  python -m voicestudio <command>   (or just `voicestudio` once installed)"""
import argparse, sys
from pathlib import Path
from . import voices
from .config import CFG


def main(argv=None):
    ap = argparse.ArgumentParser(prog="voicestudio", description="Train your own Piper voice from recordings.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="start the web page")
    sub.add_parser("doctor", help="check the GPU and tools")
    sub.add_parser("list", help="list voices")
    p = sub.add_parser("new", help="create a voice"); p.add_argument("name"); p.add_argument("--base", default="auto", help="auto (matched to your recordings) or: ryan, amy, lessac, joe, john"); p.add_argument("--folder", required=True, help="the folder with your recordings")
    p = sub.add_parser("prep", help="recordings -> dataset"); p.add_argument("name"); p.add_argument("--top-db", type=float, default=None, help="silence threshold (default: chosen automatically for each recording)")
    p = sub.add_parser("speakers", help="find who is speaking in the recordings"); p.add_argument("name"); p.add_argument("--num", type=int, default=None, help="how many people talk (default: detect)")
    p = sub.add_parser("keep", help="choose which people to train on"); p.add_argument("name"); p.add_argument("ids", nargs="*", type=int); p.add_argument("--all", action="store_true", help="use everyone (no filtering)")
    p = sub.add_parser("train", help="fine-tune (resumable)"); p.add_argument("name"); p.add_argument("epochs", nargs="?", type=int, default=1000); p.add_argument("batch", nargs="?", type=int, default=0)
    p = sub.add_parser("export", help="make the .onnx/.json and a test clip"); p.add_argument("name"); p.add_argument("--text", default=None); p.add_argument("--which", choices=["best", "accurate", "latest"], default="best", help="best = best-sounding by quality score (default)")
    p = sub.add_parser("install", help="run the configured install step"); p.add_argument("name")
    p = sub.add_parser("clean", help="free disk space: keep only the best and latest training checkpoints"); p.add_argument("name")
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
        if a.base != "auto": base.ensure(a.base)
        folder = Path(a.folder).expanduser()
        if not folder.is_dir(): sys.exit(f"Not a folder: {folder}")
        voices.write(n, base=a.base, source_dir=str(folder.resolve()))
        print(f"Created '{n}', reading recordings from {folder.resolve()}"); return
    n = CFG.check_name(a.name)
    if not voices.exists(n): sys.exit(f"No voice '{n}'. Create it first:  voicestudio new {n}")
    if a.cmd == "prep":
        from . import prep; prep.run(n, top_db=a.top_db)
    elif a.cmd == "speakers":
        from . import speakers; speakers.run(n, a.num)
    elif a.cmd == "keep":
        from . import speakers
        if a.all or not a.ids: voices.write(n, keep_people=[]); print("Using everyone in the recordings.")
        else: speakers.keep(n, a.ids); print(f"Training on person(s) {a.ids} only.")
    elif a.cmd == "train":
        from . import train; train.run(n, a.epochs, a.batch)
    elif a.cmd == "export":
        from . import export; export.run(n, a.text or export.DEFAULT_TEXT, "last" if a.which == "latest" else a.which)
    elif a.cmd == "install":
        from . import install; install.run(n)
    elif a.cmd == "clean":
        from . import train; print(f"Freed {train.prune(n):.1f} GB.")
    elif a.cmd == "delete":
        voices.delete(n); print(f"Deleted '{n}'.")
