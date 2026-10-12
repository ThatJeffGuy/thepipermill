"""Start a voice from a dataset you already have: a folder or .zip with a metadata.csv and the audio clips.

Reads  metadata.csv  (lines like  clip.wav|what is said , or LJSpeech style  id|text|text ), finds each clip next to it (in wavs/ or beside the csv),
converts it to 22.05 kHz mono, evens out the loudness, and writes the same files that Prepare writes. So the Train step can follow directly.
"""
import shutil, tempfile, zipfile
from pathlib import Path
from . import audio, prep, textnorm, voices
from .config import CFG
from .progress import Progress

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")


def safe_extract(zf, dest):
    dest = Path(dest).resolve()
    for m in zf.infolist():
        target = (dest / m.filename).resolve()
        if dest not in target.parents and target != dest: raise SystemExit(f"The zip holds an unsafe path ({m.filename}).")
    zf.extractall(dest)


def find_dataset(base):
    cands = [base / "metadata.csv"] + sorted(base.glob("*/metadata.csv"))
    md = next((c for c in cands if c.exists()), None)
    if not md: raise SystemExit("No metadata.csv found. A dataset is a folder (or zip) with a metadata.csv and the audio clips next to it, usually in a wavs/ folder.")
    return md, (md.parent / "wavs") if (md.parent / "wavs").is_dir() else md.parent


def run(name, src, log=print):
    import soundfile as sf
    src = Path(src).expanduser(); tmp = None
    try:
        if src.is_file() and src.suffix.lower() == ".zip":
            tmp = tempfile.mkdtemp(prefix="import-", dir=CFG.data); safe_extract(zipfile.ZipFile(src), tmp); base = Path(tmp)
        elif src.is_dir(): base = src
        else: raise SystemExit("Choose a folder, or a .zip file, that holds the dataset.")
        md, wavdir = find_dataset(base)
        rows = []
        for line in md.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split("|")
            if len(parts) < 2 or not parts[0].strip(): continue
            rows.append((parts[0].strip(), (parts[2] if len(parts) >= 3 and parts[2].strip() else parts[1]).strip()))
        if not rows: raise SystemExit("metadata.csv has no lines like  clip.wav|what is said.")
        root = CFG.data / name; wavs = root / "wavs"; shutil.rmtree(root, ignore_errors=True); wavs.mkdir(parents=True)
        prog = Progress(name, "import"); out, skipped, n = [], [], 0
        with tempfile.TemporaryDirectory() as td:
            for i, (f, text) in enumerate(rows, 1):
                prog.update("Converting the clips", i - 1, len(rows), f, frac=i / len(rows))
                a = next((p for p in [wavdir / f] + [wavdir / (f + e) for e in AUDIO_EXT] + [(wavdir / f).with_suffix(e) for e in AUDIO_EXT] if p.is_file()), None)
                if not a: skipped.append(f"{f}: audio file not found"); continue
                try:
                    tmpw = Path(td) / "c.wav"; secs = audio.decode(a, tmpw, prep.SR, log=lambda *x: None); y, _ = sf.read(tmpw, dtype="float32")
                except (audio.AudioError, RuntimeError, OSError) as e: skipped.append(f"{f}: {e}"); continue
                if not 0.8 <= secs <= 20: skipped.append(f"{f}: {secs:.1f} s is outside 0.8 to 20 s"); continue
                t = textnorm.normalize(text, CFG["espeak_voice"]).replace("|", " ")
                if len(t) < 3: skipped.append(f"{f}: no text"); continue
                n += 1; fn = f"{name}_{n:05d}.wav"; sf.write(wavs / fn, prep.level(y), prep.SR, subtype="PCM_16"); out.append(f"{fn}|{t}")
        if not out: raise SystemExit("No usable clips were found" + (": " + skipped[0] if skipped else "."))
        (root / "metadata.csv").write_text("\n".join(out) + "\n", encoding="utf-8"); (root / "metadata.orig.csv").write_text("\n".join(out) + "\n", encoding="utf-8")
        voices.write(name, imported=True, words_seen=False)
        mins = sum(sf.info(wavs / l.split("|")[0]).duration for l in out) / 60
        log(f"\nDone: {len(out)} clips ({mins:.1f} min) imported" + (f", {len(skipped)} skipped" if skipped else "") + ".")
        for m in skipped[:20]: log(f"  - {m}")
        log("Next: check the words, then train.")
        prog.finish(f"{len(out)} clips")
    finally:
        if tmp: shutil.rmtree(tmp, ignore_errors=True)
