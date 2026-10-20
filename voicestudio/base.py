"""Base checkpoints: the trained Piper voices we start from, so you need minutes of audio, not days.

Downloaded from https://huggingface.co/datasets/rhasspy/piper-checkpoints and converted to a cleaned copy
the current trainer accepts (loaded in PyTorch's safe weights-only mode; nothing is executed).
Each base voice has its own dataset licence: it is shown when you create a voice and saved next to the file.
"""
import json, urllib.parse, urllib.request
from pathlib import Path
from .config import CFG

HF = "https://huggingface.co/datasets/rhasspy/piper-checkpoints"
# key -> (label, path in the checkpoints dataset)   (only 'medium' quality works with the default settings)
BASES = {
    "ryan":   ("Male (Ryan)",     "en/en_US/ryan/medium"),
    "amy":    ("Female (Amy)",    "en/en_US/amy/medium"),
    "lessac": ("Female (Lessac)", "en/en_US/lessac/medium"),
    "joe":    ("Male (Joe)",      "en/en_US/joe/medium"),
    "john":   ("Male (John)",     "en/en_US/john/medium"),
}
# hyper-parameters the current trainer understands (old checkpoints carry many obsolete ones)
KEEP = {"batch_size", "betas", "c_kl", "c_mel", "dataset", "eps", "filter_channels", "filter_length", "gin_channels", "grad_clip",
        "hidden_channels", "hop_length", "init_lr_ratio", "inter_channels", "kernel_size", "learning_rate", "lr_decay", "mel_channels",
        "mel_fmax", "mel_fmin", "n_heads", "n_layers", "n_layers_q", "num_speakers", "num_symbols", "p_dropout", "resblock",
        "resblock_dilation_sizes", "resblock_kernel_sizes", "sample_rate", "segment_size", "upsample_initial_channel",
        "upsample_kernel_sizes", "upsample_rates", "use_sdp", "use_spectral_norm", "warmup_epochs", "win_length"}


AUTO = "auto"
MALE, FEMALE = "ryan", "amy"      # what "automatic" chooses between
SPLIT_HZ = 165                    # typical speaking pitch: men ~85-155 Hz, women ~165-255 Hz


def path(key): return CFG.base / f"{key}-medium-clean.ckpt"


def pick_for(name, log=print):
    """Choose the closer starting voice by measuring the average pitch of the prepared clips."""
    import random, numpy as np, soundfile as sf, librosa
    from . import voices
    said = voices.read(name).get("voice_type")                       # you told us: no need to guess from the audio
    if said in ("male", "female"):
        key = FEMALE if said == "female" else MALE
        log(f"You said this is a {said} voice -> starting from the {said} base voice ({key})."); return key
    wavs = sorted((CFG.data / name / "wavs").glob("*.wav"))
    if not wavs: return MALE
    random.Random(0).shuffle(wavs); f0s = []
    for w in wavs[:25]:
        y, sr = sf.read(w, dtype="float32")
        f0, voiced, _ = librosa.pyin(y[: sr * 6], fmin=60, fmax=400, sr=sr, frame_length=2048)
        f0s += [v for v, ok in zip(f0, voiced) if ok and not np.isnan(v)]
    if len(f0s) < 50: return MALE
    hz = float(np.median(f0s)); key = FEMALE if hz >= SPLIT_HZ else MALE
    log(f"Average pitch of your recordings: {hz:.0f} Hz -> starting from the {'female' if key == FEMALE else 'male'} base voice ({key}).")
    return key


def info(key):
    """{'epoch': int, 'license': str} for a prepared base voice, else None."""
    try: return json.loads(path(key).with_suffix(".json").read_text())
    except (OSError, ValueError): return None


def ensure(key, log=print):
    if key == AUTO: raise ValueError("pick_for() decides 'auto' once the recordings are prepared")
    if key not in BASES: raise ValueError(f"unknown base voice '{key}' (choose from {', '.join(BASES)})")
    out = path(key)
    if out.exists() and info(key): return out
    import pathlib, torch
    sub = BASES[key][1]
    tree = json.load(urllib.request.urlopen(f"{HF.replace('huggingface.co/', 'huggingface.co/api/')}/tree/main/{sub}", timeout=60))
    ck = [e["path"] for e in tree if e["path"].endswith(".ckpt")]
    if not ck: raise RuntimeError(f"no checkpoint found for {key}")
    lic = ""
    for e in tree:
        if e["path"].endswith("MODEL_CARD"):
            lic = urllib.request.urlopen(f"{HF}/resolve/main/{e['path']}", timeout=60).read().decode(errors="ignore")
    log(f"Downloading base voice '{key}' ({ck[0].split('/')[-1]}) ...")
    CFG.ensure_dirs(); raw = CFG.base / f"{key}.download"
    with urllib.request.urlopen(f"{HF}/resolve/main/{urllib.parse.quote(ck[0])}", timeout=120) as r, open(raw, "wb") as f:
        n = 0
        while chunk := r.read(1 << 22):
            f.write(chunk); n += len(chunk)
    allowed = [pathlib.PosixPath, pathlib.PurePosixPath]
    with torch.serialization.safe_globals(allowed):                       # still safe mode; only Path objects are extra
        c = torch.load(raw, weights_only=True, map_location="cpu")

    def clean(o):
        if isinstance(o, pathlib.PurePath): return str(o)
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return type(o)(clean(v) for v in o)
        return o
    c = clean(c)
    c["hyper_parameters"] = {k: v for k, v in c.get("hyper_parameters", {}).items() if k in KEEP}
    torch.save(c, out); raw.unlink(missing_ok=True)
    torch.load(out, weights_only=True, map_location="cpu")                # must load in safe mode
    out.with_suffix(".json").write_text(json.dumps({"epoch": int(c["epoch"]), "license": lic.strip()}))
    log(f"Base voice ready (trained to epoch {c['epoch']}).")
    return out
