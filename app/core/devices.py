"""Which torch device Kiln runs on, and what that device can do.

Three devices, in order of preference: an NVIDIA GPU (``cuda``), Apple's Metal
GPU on Apple Silicon (``mps``), and the CPU. Everything that picks a device or
frees one goes through here, so the order lives in one place.

``KILN_DEVICE`` overrides the choice (``cpu``, ``mps`` or ``cuda``). It exists
for the Mac: MPS is the youngest of the three backends, and a model that
misbehaves on it can still be run on the CPU without editing anything.
"""
import os

from utils import platform_mac

_DEVICES = ("cuda", "mps", "cpu")


def _torch():
    import torch  # noqa: PLC0415

    return torch


def mps_available() -> bool:
    try:
        mps = getattr(_torch().backends, "mps", None)
        return bool(mps is not None and mps.is_available())
    except Exception:  # noqa: BLE001
        return False


def best_device() -> str:
    forced = (os.environ.get("KILN_DEVICE") or "").strip().lower()
    if forced in _DEVICES:
        return forced
    torch = _torch()
    if torch.cuda.is_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def empty_cache(device: str | None = None):
    """Hand cached allocator memory back, on whichever GPU is in use."""
    try:
        torch = _torch()
        kind = str(device or best_device()).split(":")[0]
        if kind == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif kind == "mps" and mps_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _possible_devices() -> set[str]:
    """Devices this *platform* could ever offer, whatever is installed.

    Deliberately not "what torch sees right now": off the Mac, whether CUDA
    works is the xurdif trainer's own question, asked of the interpreter it
    launches, with a diagnosis of the driver and wheel when it does not.
    Refusing earlier, here, would replace that diagnosis with a vaguer one.
    """
    if not platform_mac.is_mac():
        return {"cuda", "cpu"}
    return {"mps", "cpu"} if platform_mac.is_apple_silicon() else {"cpu"}


def train_block_reason(backend) -> str | None:
    """Why ``backend`` cannot train on this machine, or None when it can."""
    if platform_mac.is_mac() and not platform_mac.is_apple_silicon():
        if platform_mac.under_rosetta():
            return ("Python is running under Rosetta, so PyTorch cannot use this Mac's GPU. "
                    "Reinstall Kiln with an arm64 Python from python.org to train.")
        return "Training on a Mac needs Apple Silicon."
    if set(backend.capabilities.train_devices) & _possible_devices():
        return None
    if platform_mac.is_mac():
        return (f"{backend.name} models train on NVIDIA GPUs only. On this Mac, train a "
                "Diffusers model, or re-home a tinyunet_with_attention3 model into "
                "Diffusers format under Prepare ▸ Models ▸ Re-home a model.")
    return f"{backend.name} cannot train on this machine."
