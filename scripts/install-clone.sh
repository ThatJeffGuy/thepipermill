#!/usr/bin/env bash
# Installs the optional "fast path" (voice cloning) into its OWN environment, ./.venv-clone, so it can never disturb training.
# About 6 GB of downloads (PyTorch, the OmniVoice cloner, the F5-TTS alternative, the speaker-voiceprint model); the cloners' weights (about 3.3 GB for OmniVoice, 1.3 GB for F5) download on first use.
# OmniVoice is Apache-2.0; F5-TTS's weights are licensed for non-commercial use. Only clone voices of people who have agreed to it.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null 2>&1 || { echo "This needs 'uv' (https://docs.astral.sh/uv/). Install it, then run this again."; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || echo "Note: no NVIDIA driver found; cloning will run on the CPU (slow)."
echo "== Creating .venv-clone (Python 3.11)"
uv venv --python 3.11 .venv-clone -q
pipi() { uv pip install -q --python .venv-clone/bin/python "$@"; }
echo "== Installing PyTorch"
pipi torch torchaudio
echo "== Installing the cloner and the voiceprint model"
pipi omnivoice f5-tts speechbrain soundfile librosa
# pyarrow can crash on import on some systems and is only used by F5-TTS's own training code, which we never run.
uv pip uninstall -q --python .venv-clone/bin/python pyarrow 2>/dev/null || true
echo "== Checking"
.venv-clone/bin/python - <<'PY'
import sys, types
sys.modules["datasets"] = types.ModuleType("datasets"); sys.modules["datasets"].Dataset = object; sys.modules["datasets"].load_from_disk = None
import torch, omnivoice, f5_tts, speechbrain
print("torch", torch.__version__, "| GPU:", torch.cuda.is_available())
PY
echo "Done. The fast path is ready."
