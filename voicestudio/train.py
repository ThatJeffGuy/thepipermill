"""Fine-tune a voice on the GPU, resumable.

Starts from the base voice you picked, or from this voice's newest checkpoint if it already has one.
Lightning keeps counting epochs from the base checkpoint, so the stopping point is (base epoch + extra_epochs).
"""
import hashlib, os, re, shutil, subprocess, sys
from . import base, gpu, voices
from .config import CFG


def all_ckpts(name):
    """Every checkpoint of this voice. Resuming training starts a new run folder, so look in all of them."""
    return list((CFG.runs / name).glob("lightning_logs/version_*/checkpoints/*.ckpt"))


def newest_ckpt(name):
    """What to resume from: the most recent last.ckpt (Piper's latest full state), else the most recently written checkpoint."""
    cks = all_ckpts(name)
    pool = [p for p in cks if p.name == "last.ckpt"] or cks
    return max(pool, key=lambda p: p.stat().st_mtime) if pool else None


def _scored(name, key):
    out = []
    for p in all_ckpts(name):
        m = re.search(rf"{key}=([0-9.]+)\.ckpt$", p.name)
        if m: out.append((float(m.group(1)), p))
    return out


def pick(name, which="best"):
    """(checkpoint, description). Piper keeps its best checkpoints by quality score (val_mos) and by accuracy (val_mel):
    the best-sounding one is usually better than simply the latest."""
    mos, mel = _scored(name, "val_mos"), _scored(name, "val_mel")
    if which == "best" and mos:
        sc, p = max(mos); return p, f"best-sounding version (quality score {sc:.2f})"
    if which in ("best", "accurate") and mel:
        sc, p = min(mel); return p, f"most accurate version (mel error {sc:.3f})"
    return newest_ckpt(name), "latest version"


def prune(name):
    """Free disk space: keep only the best-sounding, the most accurate and the latest checkpoint. Returns GB freed."""
    keep = {p for p in (newest_ckpt(name), pick(name, "best")[0], pick(name, "accurate")[0]) if p}
    freed = 0
    for p in all_ckpts(name):
        if p not in keep: freed += p.stat().st_size; p.unlink()
    for d in (CFG.runs / name).glob("lightning_logs/version_*/checkpoints"):      # tidy empty run folders
        if not any(d.iterdir()): d.rmdir()
    return freed / 2**30


def _dataset_fingerprint(name):
    h = hashlib.sha256()
    md = CFG.data / name / "metadata.csv"; h.update(md.read_bytes())
    for line in md.read_text(encoding="utf-8").splitlines():
        w = CFG.data / name / "wavs" / line.split("|", 1)[0]
        if w.exists(): h.update(f"{w.name}{w.stat().st_size}".encode())
    return h.hexdigest()


def _check_cache(name):
    """Piper caches processed clips by row number + text. If the dataset changed, the cache could hold the wrong audio: clear it."""
    fp, f = _dataset_fingerprint(name), CFG.cache / name / ".dataset"
    if f.exists() and f.read_text() != fp:
        print("The prepared clips or transcripts changed since the last run: refreshing Piper's cache.", flush=True)
        shutil.rmtree(CFG.cache / name, ignore_errors=True)
    (CFG.cache / name).mkdir(parents=True, exist_ok=True); f.write_text(fp)


def run(name, extra_epochs=1000, batch=0):
    gpu.check()
    m = voices.read(name)
    if not (CFG.data / name / "metadata.csv").exists(): raise SystemExit("Prepare the recordings first.")
    key = m.get("base", base.AUTO)
    if key == base.AUTO:
        key = m.get("base_used") or base.pick_for(name, log=lambda t: print(t, flush=True))
        voices.write(name, base_used=key)           # remember it so resuming uses the same starting voice
    b = base.ensure(key)
    b_epoch = (base.info(key) or {}).get("epoch", 0)
    batch = batch or CFG["batch_size"] or gpu.default_batch()
    run_dir = CFG.runs / name; run_dir.mkdir(parents=True, exist_ok=True); _check_cache(name)
    ckpt = newest_ckpt(name)
    print(f"{'resuming from ' + str(ckpt) if ckpt else 'starting from base voice ' + key}; batch size {batch}; "
          f"training to epoch {b_epoch + extra_epochs} (base is at {b_epoch})", flush=True)
    cmd = [sys.executable, "-m", "piper.train", "fit",
           "--data.voice_name", name,
           "--data.csv_path", str(CFG.data / name / "metadata.csv"),
           "--data.audio_dir", str(CFG.data / name / "wavs"),
           "--model.sample_rate", "22050",
           "--data.espeak_voice", CFG["espeak_voice"],
           "--data.cache_dir", str(CFG.cache / name),
           "--data.config_path", str(run_dir / "config.json"),
           "--data.batch_size", str(batch),
           "--data.num_workers", str(max(2, min(6, (os.cpu_count() or 4) // 2))),
           "--trainer.accelerator", "gpu", "--trainer.devices", "1",
           "--trainer.max_epochs", str(b_epoch + extra_epochs),
           "--trainer.default_root_dir", str(run_dir),
           "--trainer.log_every_n_steps", "10",
           "--ckpt_path", str(ckpt or b)]
    sys.exit(subprocess.call(cmd))
