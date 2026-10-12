#!/usr/bin/env bash
# Voice Studio installer. Linux + an NVIDIA GPU. Creates ./.venv and downloads the Piper trainer into ./vendor.
set -euo pipefail
cd "$(dirname "$0")"

PIPER_REPO="https://github.com/OHF-Voice/piper1-gpl.git"
PIPER_COMMIT="35b0f1e"          # the version this was tested with (a newer one may need small changes)

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1  ($2)"; exit 1; }; }

say "Checking your system"
need git "install git"; need ffmpeg "install ffmpeg"; need gcc "install build tools, e.g. sudo apt install build-essential"
need g++ "install build tools, e.g. sudo apt install build-essential"
command -v nvidia-smi >/dev/null 2>&1 || { echo "No NVIDIA driver found (nvidia-smi). Training needs an NVIDIA GPU."; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if command -v uv >/dev/null 2>&1; then
  say "Creating the Python 3.11 environment (uv)"
  uv venv --python 3.11 .venv
  pipi() { uv pip install --python .venv/bin/python "$@"; }
else
  PY=""; for c in python3.11 python3.12 python3; do command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys;sys.exit(not (3,11)<=sys.version_info[:2]<=(3,12))' && { PY="$c"; break; }; done
  [ -n "$PY" ] || { echo "Need Python 3.11 or 3.12 (or install 'uv': https://docs.astral.sh/uv/)."; exit 1; }
  say "Creating the Python environment ($PY)"
  "$PY" -m venv .venv
  pipi() { .venv/bin/python -m pip install "$@"; }
  pipi -q -U pip
fi

say "Getting the Piper trainer ($PIPER_COMMIT)"
mkdir -p vendor
[ -d vendor/piper1-gpl/.git ] || git clone "$PIPER_REPO" vendor/piper1-gpl
git -C vendor/piper1-gpl checkout -q "$PIPER_COMMIT"

say "Installing PyTorch (this is the big download)"
pipi torch torchaudio cmake ninja scikit-build setuptools wheel cython

say "Installing the trainer"
pipi -e "./vendor/piper1-gpl[train]"
export PATH="$PWD/.venv/bin:$PATH"      # so the build tools (cython, cmake, ninja) are found

say "Building the speech-engine pieces (takes a few minutes)"
healthy() { python - <<'PY' >/dev/null 2>&1
from piper import espeakbridge
import piper.train.vits.monotonic_align
PY
}
for attempt in 1 2 3; do
  ( cd vendor/piper1-gpl && python setup.py build_ext --inplace && ./build_monotonic_align.sh ) && healthy && break
  echo "Build attempt $attempt did not finish; trying again..."
done
healthy || { echo "ERROR: could not build the espeak-ng bridge / alignment code. Scroll up for the compiler error."; exit 1; }

# Newer PyTorch's default exporter fails on this model; the older TorchScript exporter works.
.venv/bin/python - <<'PY'
from pathlib import Path
p = Path("vendor/piper1-gpl/src/piper/train/export_onnx.py"); s = p.read_text()
if "dynamo=False" not in s:
    s = s.replace("        verbose=False,\n", "        verbose=False,\n        dynamo=False,\n", 1); p.write_text(s); print("patched export_onnx.py")
PY

say "Installing Voice Studio and its audio tools"
pipi -c constraints.txt transformers accelerate soundfile librosa numpy onnx onnxscript onnxruntime pyannote.audio num2words
pipi -e .

say "Checking the GPU"
.venv/bin/voicestudio doctor

cat <<DONE

Done. Start it with:   ./run.sh      then open  http://127.0.0.1:8765
Optional settings:      cp config.example.toml ~/.config/voice-studio/config.toml
Start automatically:    scripts/install-service.sh
DONE
