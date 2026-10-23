"""The voice separator: remove music, sound effects and background noise, keep only the voices (Demucs "htdemucs", vocals stem).

Used by the Prepare step (master.py), which cleans every recording once, whole, before anything else reads it.
Your recordings are never changed. The Demucs model (about 80 MB) downloads the first time it is used."""
SR = 22050                  # the sample rate the cleaned audio is written at; Demucs itself runs at 44.1 kHz inside
DEMUCS_SR = 44100
CHUNK_S, OVERLAP_S = 300, 4  # a long stretch goes through in 5 minute pieces so it always fits in GPU memory
CONTEXT_S = 1.5              # a little audio on each side of a short stretch, so its edges separate as well as its middle


class Separator:
    def __init__(self, log=print):
        import torch
        try:
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
        except ImportError:
            raise SystemExit("Demucs is not installed. Run ./install.sh again (or: uv pip install --no-deps demucs einops julius lameenc openunmix).")
        self.torch, self.apply = torch, apply_model
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dev == "cpu": log("WARNING: no GPU found; removing the background will be slow.")
        log("Loading the voice separator (the first time this downloads about 80 MB)...")
        self.model = get_model("htdemucs").to(self.dev).eval(); self.vi = self.model.sources.index("vocals")

    def _piece(self, y):
        """Vocals of one piece of mono audio at SR, same length."""
        import numpy as np, librosa
        x = librosa.resample(y, orig_sr=SR, target_sr=DEMUCS_SR)
        t = self.torch.from_numpy(np.stack([x, x]))[None].to(self.dev)
        with self.torch.no_grad(): v = self.apply(self.model, t, device=self.dev, split=True, overlap=0.25, progress=False)[0, self.vi]
        v = librosa.resample(v.mean(0).cpu().numpy(), orig_sr=DEMUCS_SR, target_sr=SR)
        v = np.nan_to_num(v[: len(y)]); return np.pad(v, (0, len(y) - len(v))).astype("float32")

    def vocals(self, y):
        """Vocals of any length of mono audio at SR (long audio goes through in overlapping pieces; the shaky piece edges are dropped)."""
        import numpy as np
        n, chunk, step, half = len(y), CHUNK_S * SR, (CHUNK_S - OVERLAP_S) * SR, (OVERLAP_S // 2) * SR
        if n <= chunk: return self._piece(y)
        out, pos = np.zeros(n, "float32"), 0
        while True:
            a, b = pos, min(n, pos + chunk); last = b >= n
            v = self._piece(y[a:b]); lo = 0 if a == 0 else half; hi = (b - a) if last else (b - a) - half
            out[a + lo:a + hi] = v[lo:hi]
            if last: return out
            pos += step

    def span(self, y, a, b):
        """Vocals of y[a:b] (sample positions), using a little context from either side."""
        pad = int(CONTEXT_S * SR); c0, c1 = max(0, a - pad), min(len(y), b + pad)
        v = self.vocals(y[c0:c1]); return v[a - c0: a - c0 + (b - a)]

    def close(self):
        del self.model; self.torch.cuda.empty_cache()
