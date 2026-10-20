"""Fine-tune a voice on the GPU, resumable.

Starts from the base voice you picked, or from this voice's newest checkpoint if it already has one.
Lightning keeps counting epochs from the base checkpoint, so the stopping point is (base epoch + extra_epochs).
"""
import hashlib, json, os, re, shutil, subprocess, sys
from . import base, dataset, gpu, voices
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


def _epoch(p):
    m = re.search(r"epoch=(\d+)", p.name); return int(m.group(1)) if m else None


def ckpt_epoch(p):
    """The epoch a checkpoint was saved at. last.ckpt has no epoch in its name, so read it from the file itself."""
    e = _epoch(p)
    if p.name != "last.ckpt" and e is not None: return e
    try:
        import torch
        return int(torch.load(p, map_location="cpu", weights_only=False, mmap=True).get("epoch"))
    except Exception:
        return e


def base_epoch(name):
    """The epoch the starting voice was already at (training counts on from there)."""
    key = voices.read(name).get("base_used")
    try: return int((base.info(key) or {}).get("epoch", 0)) if key else 0
    except (TypeError, ValueError): return 0


def pick(name, which="best"):
    """(checkpoint, description). Piper keeps its best checkpoints by quality score (val_mos) and by accuracy (val_mel).
    "Best sounding" ignores the first rounds of training: there the voice still sounds like the starting voice, which can score
    well on naturalness without sounding like you. If the best scores all come from those first rounds (the score has only gone
    down since), it falls back to the most accurate version."""
    mos, mel = _scored(name, "val_mos"), _scored(name, "val_mel")
    eps = [e for e in (_epoch(p) for p in all_ckpts(name)) if e is not None]
    done = (max(eps) - base_epoch(name)) if eps else 0
    start = base_epoch(name) + max(20, int(0.1 * done))                    # skip at least the first 20 rounds (or 10%)
    if which == "best" and mos:
        late = [(sc, p) for sc, p in mos if (_epoch(p) or 0) >= start]
        if late:
            sc, p = max(late); return p, f"best-sounding version (quality score {sc:.2f}, after the first rounds)"
        if mel:
            sc, p = min(mel); return p, f"most accurate version (mel error {sc:.3f}); the quality score was highest right at the start, so it is not a useful guide here"
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


def select_clips(name, log=print):
    """The metadata file to train on. Prepare already keeps the dataset within the voice's limit, so this is normally the whole dataset.
    If you lowered the limit afterwards, a smaller subset is written to metadata.train.csv (nothing is deleted)."""
    root = CFG.data / name; md = root / "metadata.csv"; lim = dataset.limit(name); out = root / "metadata.train.csv"
    rows = [l for l in md.read_text(encoding="utf-8").splitlines() if "|" in l]
    durs = [dataset._dur(root, r.partition("|")[0]) for r in rows]
    if sum(durs) / 60 <= lim: out.unlink(missing_ok=True); return md
    keep = dataset.choose(rows, durs, dataset.read_json(CFG.runs / name / "clipscores.json", {}).get("scores", {}),
                          set(dataset.read_json(CFG.runs / name / "prep.approved.json", [])), lim * 60)
    out.write_text("\n".join(rows[i] for i in sorted(keep)) + "\n", encoding="utf-8")
    log(f"Using {len(keep)} of {len(rows)} clips ({sum(durs[i] for i in keep) / 60:.0f} of {sum(durs) / 60:.0f} min): the limit is {lim} minutes.")
    return out


def _dataset_fingerprint(name, md):
    h = hashlib.sha256()
    h.update(md.read_bytes())
    for line in md.read_text(encoding="utf-8").splitlines():
        w = CFG.data / name / "wavs" / line.split("|", 1)[0]
        if w.exists(): h.update(f"{w.name}{w.stat().st_size}".encode())
    return h.hexdigest()


def _check_cache(name, md):
    """Piper caches processed clips by row number + text. If the dataset changed, the cache could hold the wrong audio: clear it."""
    fp, f = _dataset_fingerprint(name, md), CFG.cache / name / ".dataset"
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
    batch = batch or gpu.scaled_batch(CFG["batch_size"] or gpu.default_batch(), m.get("gpu_load") or 100)
    run_dir = CFG.runs / name; run_dir.mkdir(parents=True, exist_ok=True); csv = select_clips(name, log=lambda t: print(t, flush=True)); _check_cache(name, csv)
    ckpt = newest_ckpt(name)
    print(f"{'resuming from ' + str(ckpt) if ckpt else 'starting from base voice ' + key}; batch size {batch}; "
          f"training to epoch {b_epoch + extra_epochs} (base is at {b_epoch})", flush=True)
    cur = ckpt_epoch(ckpt) if ckpt else None
    if cur is not None and cur >= b_epoch + extra_epochs:      # the trainer would stop with a confusing error: say what is going on instead
        done = cur - b_epoch
        print(f"This voice has already trained {done} rounds, which is at or past the length you chose ({extra_epochs}). "
              f"Choose a length above {done} rounds to carry on, or start the training over from the base voice "
              f"(Advanced, Voice settings).", flush=True)
        sys.exit(2)
    cmd = [sys.executable, "-m", "piper.train", "fit",
           "--data.voice_name", name,
           "--data.csv_path", str(csv),
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
    if CFG["train_precision"]:
        cmd += ["--trainer.precision", str(CFG["train_precision"])]; print(f"training precision: {CFG['train_precision']}", flush=True)
    sys.exit(subprocess.call(cmd))
