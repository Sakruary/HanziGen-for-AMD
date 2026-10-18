"""
Hardware utilities shared by every HanziGen entry point.

This module is the single place that decides which accelerator the project
runs on. It no longer assumes an NVIDIA GPU: CUDA, AMD ROCm/HIP, AMD ZLUDA,
Windows DirectML, Apple MPS and CPU are all supported.

Use :func:`select_device` to obtain a ``torch.device`` and
:func:`autocast_context` / :func:`make_grad_scaler` to obtain mixed precision
objects that are valid on the selected backend.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from .backends import (
    AMP_DTYPE_ENV_VAR,
    BACKEND_ENV_VAR,
    HardwareBackend,
    amp_requires_grad_scaler,
    autocast_context,
    autocast_dtype,
    configure_stdio,
    describe_device,
    get_device_info,
    is_accelerator,
    is_amd,
    make_grad_scaler,
    no_autocast,
    normalize_device,
    resolve_amp_dtype,
    scaler_is_enabled,
    supports_amp,
    supports_bfloat16,
    supports_grad_scaler,
)

__all__ = [
    "AMP_DTYPE_ENV_VAR",
    "BACKEND_ENV_VAR",
    "HardwareBackend",
    "amp_requires_grad_scaler",
    "autocast_context",
    "autocast_dtype",
    "configure_miopen_cache",
    "configure_stdio",
    "describe_device",
    "enable_fast_math",
    "get_device_info",
    "is_accelerator",
    "is_amd",
    "load_state_dict",
    "make_grad_scaler",
    "no_autocast",
    "normalize_device",
    "print_backend_banner",
    "print_model_params",
    "resolve_amp_dtype",
    "resolve_amp_settings",
    "save_state_dict",
    "scaler_is_enabled",
    "select_device",
    "supports_amp",
    "supports_bfloat16",
    "supports_grad_scaler",
]


def select_device(
    device: str | torch.device | None = None,
) -> torch.device:
    """
    Select the optimal device for PyTorch operations.

    Accepts, in addition to the usual PyTorch strings:

    * ``None`` / ``"auto"`` — auto-detect, preferring CUDA/ROCm/ZLUDA, then MPS,
      then DirectML, then CPU;
    * ``"rocm"`` / ``"hip"`` / ``"amd"`` — force the HIP path (torch device ``cuda``);
    * ``"zluda"`` — force the ZLUDA path (torch device ``cuda``);
    * ``"directml"`` / ``"dml"`` — force the Windows DirectML path
      (torch device ``privateuseone``);
    * ``"cuda"``, ``"cuda:1"``, ``"mps"``, ``"cpu"`` — standard PyTorch strings.

    The ambient override from the ``HANZIGEN_BACKEND`` environment variable is
    honoured when ``device`` is ``None``.
    """
    resolved, _ = normalize_device(device)
    return resolved


def resolve_amp_settings(
    device: torch.device | str | None,
    use_amp: bool = False,
) -> tuple[bool, str | None]:
    """
    Clamp a mixed precision request to what the selected backend can do.

    Returns:
        ``(effective_use_amp, warning)`` where ``warning`` is ``None`` when the
        request could be honoured as-is.
    """
    if not use_amp:
        return False, None

    info = get_device_info(device)

    if not supports_grad_scaler(device):
        return (
            False,
            f"[warn] Mixed precision (USE_AMP) is not supported on "
            f"{info.backend.display_name}; training will run in float32.",
        )

    if info.backend is HardwareBackend.ZLUDA:
        return (
            True,
            "[warn] ZLUDA detected: float16 autocast support depends on the ZLUDA "
            "build. If you see NaNs or a device-side assert, set USE_AMP=false.",
        )

    return True, None


def enable_fast_math(
    device: torch.device | str | None = None,
) -> None:
    """
    Enable the backend specific fast-math / TF32 settings that are safe here.

    * NVIDIA CUDA: allow TF32 for matmul and convolutions, and let cuDNN pick
      kernels by benchmarking.
    * Everyone else: only allow TF32-style reduced precision on matmul, and
      never touch ``torch.backends.cudnn``.

    Why cuDNN benchmarking is NVIDIA-only
    -------------------------------------
    ``torch.backends.cudnn.benchmark = True`` makes the convolution backend
    time every candidate kernel on the first call. On NVIDIA that is a
    well-bounded search.

    On ROCm the same attribute is *present and reports itself available* --
    it is a compatibility shim over MIOpen -- so guarding on
    ``torch.backends.cudnn.is_available()`` is not enough to keep it off AMD
    hardware. MIOpen compiles kernels through the JIT compiler rather than
    shipping a prebuilt database, so the search costs seconds of CPU per
    convolution shape on the first call instead of milliseconds. Measured on
    Windows 11 + RX 9070 XT with torch 2.10.0+rocm7.13.0a20260511, the first
    512x512 convolution took ~4 s and a single small convolution in the smoke
    test took ~42 s with a cold MIOpen cache. Steady-state throughput was
    identical either way (134 vs 135 images/s), so the search buys nothing
    here and only delays the start of training.

    Disabling it therefore costs at most a slightly suboptimal kernel choice,
    which is a far better trade than tens of seconds of dead time before the
    first batch completes.
    """
    info = get_device_info(device)

    if info.backend is HardwareBackend.CPU:
        return

    # TF32 is a matmul-level relaxation and is safe wherever PyTorch exposes
    # the flag. It is a no-op on backends that do not implement it.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:  # pragma: no cover - build dependent
        pass

    if info.backend is not HardwareBackend.CUDA:
        # Deliberately do not touch torch.backends.cudnn here; see above.
        return

    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:  # pragma: no cover - build dependent
        pass


def print_backend_banner(
    device: torch.device | str | None = None,
    use_amp: bool = False,
) -> None:
    """
    Print a short, complete summary of the active compute configuration.
    """
    info = get_device_info(device)

    print("[compute] backend summary")
    print(f"   Backend          : {info.backend.display_name} ({info.backend.value})")
    print(f"   Vendor           : {info.backend.vendor}")
    print(f"   Device           : {info.name}")
    print(f"   Torch device     : {info.device}")
    print(f"   PyTorch          : {torch.__version__}")
    if info.is_rocm_build:
        print(f"   HIP runtime      : {torch.version.hip}")
    elif info.is_cuda_build:
        print(f"   CUDA runtime     : {torch.version.cuda}")
    if info.compute_capability:
        print(f"   Architecture     : {info.compute_capability}")
    if info.total_memory:
        print(f"   Memory           : {info.total_memory / 1024**3:.2f} GB")
    if use_amp:
        amp_dtype = resolve_amp_dtype(device)
        dtype_label = "bf16" if amp_dtype == torch.bfloat16 else "fp16"
        print(f"   Mixed precision  : enabled ({dtype_label})")
    else:
        print(f"   Mixed precision  : disabled")

    for note in info.notes:
        print(f"   Note             : {note}")

    if info.backend is HardwareBackend.CPU:
        print(
            "   Warning          : no GPU backend was detected; training will be "
            "extremely slow. See docs/AMD_SETUP.md or docs/NVIDIA_SETUP.md."
        )


def load_state_dict(
    path: str | "os.PathLike[str]",
) -> dict:
    """
    Read a checkpoint into ordinary CPU tensors.

    Checkpoints are always loaded onto the CPU rather than straight onto the
    target device:

    * it is the portable choice, since a checkpoint may have been written on a
      different backend (a CUDA-trained file must load on ROCm, and vice versa);
    * it avoids backend specific restore paths. ``torch.load(map_location=...)``
      with a ``torch.device`` breaks on DirectML, whose storage restore only
      accepts an integer device index, and raising the need for a special case
      inside every caller is worse than resolving it once here.

    Callers move the model to the device afterwards, which is what actually
    matters. ``weights_only=True`` is kept because every checkpoint written by
    this project is a pure state dict.
    """
    return torch.load(path, map_location="cpu", weights_only=True)


def save_state_dict(
    module: nn.Module,
    path: str | "os.PathLike[str]",
) -> None:
    """
    Save a module's weights in a device-independent, safely loadable form.

    ``torch.save(module.state_dict(), path)`` writes the tensors exactly as
    they live on the device. On CUDA that is fine, but on DirectML the tensors
    are ``privateuseone`` tensors, which pickle as
    ``torch._utils._rebuild_device_tensor_from_numpy``. Loading such a file
    with the default ``weights_only=True`` then fails outright, so a VQ-VAE
    checkpoint produced on DirectML could not be consumed by LDM training.

    Moving everything to CPU first produces an ordinary checkpoint: portable
    between backends, smaller on disk, and loadable with ``weights_only=True``.
    """
    state = {
        key: (value.detach().cpu() if isinstance(value, torch.Tensor) else value)
        for key, value in module.state_dict().items()
    }
    torch.save(state, path)


#: Directory (relative to the current working directory) used for tool caches
#: that some backends insist on, when the user's home directory is not usable.
LOCAL_CACHE_DIRNAME = ".cache"


def configure_miopen_cache(cache_root: str | "os.PathLike[str] | None" = None) -> str | None:
    """
    Point MIOpen at a writable cache directory and return it.

    Why this exists
    ---------------
    MIOpen compiles every convolution kernel on first use and stores the result
    in ``~/.miopen``. Two things go wrong when that directory is not writable
    (restricted profile, corporate policy, read-only home, sandboxed shell):

    * the SQLite find-database fails to open, so *every* convolution re-runs the
      whole kernel search, and
    * MIOpen eventually raises ``miopenStatusInternalError``.

    Measured on Windows 11 + RX 9070 XT (torch 2.10.0+rocm7.13.0a20260511), one
    512x512 convolution at batch 24 took **46.6 s** with an unusable cache and
    **0.002 s** once the cache worked -- the difference between a training run
    that starts and one that appears to hang.

    The fix is only the two environment variables, and MIOpen reads them when it
    is first used, so setting them before the first convolution is enough. An
    explicit user setting always wins.

    Returns:
        The directory that was configured, or ``None`` when nothing was needed
        or nothing writable could be found.
    """
    if os.environ.get("MIOPEN_USER_DB_PATH") and os.environ.get("MIOPEN_CUSTOM_CACHE_DIR"):
        return None  # the user already decided

    root = Path(cache_root) if cache_root is not None else Path.cwd() / LOCAL_CACHE_DIRNAME
    database_dir = root / "miopen" / "db"
    kernel_dir = root / "miopen" / "cache"

    try:
        database_dir.mkdir(parents=True, exist_ok=True)
        kernel_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Not writable: nothing we can do from here. Leave the defaults alone so
        # the failure message still points at the real cause.
        return None

    configured: list[str] = []
    if not os.environ.get("MIOPEN_USER_DB_PATH"):
        os.environ["MIOPEN_USER_DB_PATH"] = str(database_dir)
        configured.append("MIOPEN_USER_DB_PATH")
    if not os.environ.get("MIOPEN_CUSTOM_CACHE_DIR"):
        os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = str(kernel_dir)
        configured.append("MIOPEN_CUSTOM_CACHE_DIR")

    return str(root) if configured else None


def print_model_params(model: nn.Module) -> None:
    """
    Print the total number of parameters in a PyTorch model.
    """
    params = sum(p.numel() for p in model.parameters())

    if params >= 1e9:
        formatted_params = f"{params / 1e9:.2f}B"
    elif params >= 1e6:
        formatted_params = f"{params / 1e6:.2f}M"
    elif params >= 1e3:
        formatted_params = f"{params / 1e3:.2f}K"
    else:
        formatted_params = str(params)

    print(f"[model] total parameters: {formatted_params}")
