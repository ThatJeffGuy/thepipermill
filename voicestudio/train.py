"""Fine-tune a voice on the GPU, resumable.

Starts from the base voice you picked, or from this voice's newest checkpoint if it already has one.
Lightning keeps counting epochs from the base checkpoint, so the stopping point is (base epoch + extra_epochs).
"""
import subprocess, sys
from . import base, gpu, voices
from .config import CFG


def newest_ckpt(name):
    cks = sorted((CFG.runs / name).glob("lightning_logs/version_*/checkpoints/*.ckpt"), key=lambda p: p.stat().st_mtime)
    return cks[-1] if cks else None


def run(name, extra_epochs=1000, batch=0):
    gpu.check()
    m = voices.read(name)
    if not (CFG.data / name / "metadata.csv").exists(): raise SystemExit("Prepare the recordings first.")
    key = m.get("base", "ryan"); b = base.ensure(key)
    b_epoch = (base.info(key) or {}).get("epoch", 0)
    batch = batch or CFG["batch_size"] or gpu.default_batch()
    run_dir = CFG.runs / name; run_dir.mkdir(parents=True, exist_ok=True); (CFG.cache / name).mkdir(parents=True, exist_ok=True)
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
           "--trainer.accelerator", "gpu", "--trainer.devices", "1",
           "--trainer.max_epochs", str(b_epoch + extra_epochs),
           "--trainer.default_root_dir", str(run_dir),
           "--trainer.log_every_n_steps", "10",
           "--ckpt_path", str(ckpt or b)]
    sys.exit(subprocess.call(cmd))
