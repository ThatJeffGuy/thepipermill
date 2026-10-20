"""Make sure training will really run on an NVIDIA GPU (never silently on the CPU)."""
import sys


def check(quiet=False):
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA is not available to PyTorch, so training would run on the CPU. Refusing to start.\n"
                         "Check the NVIDIA driver (nvidia-smi) and that torch was installed with CUDA support.")
    d = torch.cuda.get_device_properties(0)
    x = torch.randn(2048, 2048, device="cuda"); torch.cuda.synchronize(); (x @ x).sum().item()   # run a real kernel
    arch = f"sm_{d.major}{d.minor}"
    info = {"name": d.name, "gb": d.total_memory / 2**30, "arch": arch, "torch": torch.__version__,
            "cuda": torch.version.cuda, "arch_supported": arch in torch.cuda.get_arch_list()}
    if not quiet:
        print(f"GPU OK: {info['name']}, {info['gb']:.0f} GB, {arch} (supported by this torch: {info['arch_supported']}), "
              f"torch {info['torch']} / CUDA {info['cuda']}")
    return info


def batch_for_gb(gb):
    """The largest batch size that is safe for a card with this much memory."""
    return 8 if gb < 9 else 16 if gb < 20 else 32


def default_batch():
    return batch_for_gb(check(quiet=True)["gb"])


def scaled_batch(top, percent):
    """The batch size for a "GPU usage" setting: percent (10 to 100) of the safe size, never below 2."""
    return max(2, int(round(top * max(10, min(100, percent)) / 100))) if top > 2 else top
