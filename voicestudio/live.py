"""Keep an exported voice loaded in the web tool's memory so test clips render in a fraction of a second (no process start, no model load).
Runs on the CPU on purpose: the GPU belongs to training. The same render is available to other programs as POST /api/live with raw=true."""
import io, threading, wave
from .config import CFG

_lock = threading.Lock()
_voices = {}          # name -> (PiperVoice, onnx mtime)
_keep = set()         # names the owner wants held in memory


def _load(name):
    onnx = CFG.out / f"{name}.onnx"
    if not onnx.exists(): raise ValueError("Make a test clip once first: that exports the voice.")
    mt = onnx.stat().st_mtime_ns
    got = _voices.get(name)
    if got and got[1] == mt: return got[0]                         # still the same export
    from piper import PiperVoice
    v = PiperVoice.load(onnx, config_path=CFG.out / f"{name}.onnx.json", use_cuda=False)
    _voices[name] = (v, mt)
    return v


def keeping(name): return name in _keep


def loaded(name): return name in _voices


def set_keep(name, on):
    with _lock:
        if on: _keep.add(name); _load(name)
        else: _keep.discard(name); _voices.pop(name, None)


def render(name, text):
    """WAV bytes for the text. The model stays loaded only if keep is on."""
    with _lock:
        v = _load(name)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w: v.synthesize_wav(text, w)
        if name not in _keep: _voices.pop(name, None)
        return buf.getvalue()
