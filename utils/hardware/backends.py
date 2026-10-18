"""
Hardware backend detection for HanziGen.

This module isolates every vendor specific detail of the runtime so that the
rest of the project never has to assume an NVIDIA/CUDA machine. It supports:

``cuda``
    NVIDIA GPUs (the original target of this project).
``rocm``
    AMD GPUs driven by ROCm/HIP. PyTorch exposes HIP devices through the
    ``torch.cuda`` namespace, so the torch device type stays ``"cuda"`` while
    ``torch.version.hip`` is set. This is the recommended AMD path (native
    Linux ROCm, Windows ROCm "gfx120X-all" wheels, or WSL2).
``zluda``
    AMD GPUs on Windows through the ZLUDA CUDA translation layer. ZLUDA is
    indistinguishable from CUDA for most attribute checks, so it is detected
    through a set of heuristics and through the ``HANZIGEN_BACKEND`` override.
``directml``
    AMD/Intel GPUs on Windows through ``torch-directml``. Device type is
    ``"privateuseone"``.
``mps``
    Apple Silicon GPUs.
``cpu``
    Fallback.

The entry points of this project only need :func:`select_device`,
:func:`autocast_context` and :func:`make_grad_scaler`.
"""

from __future__ import annotations

import os
import platform
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

__all__ = [
    "AMP_DTYPE_ENV_VAR",
    "BACKEND_ENV_VAR",
    "HardwareBackend",
    "amp_requires_grad_scaler",
    "autocast_context",
    "autocast_dtype",
    "configure_stdio",
    "describe_device",
    "device_index",
    "detect_backend",
    "get_device_info",
    "is_accelerator",
    "is_amd",
    "make_grad_scaler",
    "no_autocast",
    "normalize_device",
    "reset_device_cache",
    "resolve_amp_dtype",
    "scaler_is_enabled",
    "select_device",
    "supports_amp",
    "supports_bfloat16",
    "supports_grad_scaler",
]

#: Set this environment variable to force a backend, e.g. ``HANZIGEN_BACKEND=zluda``.
#: Accepted values: ``auto``, ``cuda``, ``rocm``, ``zluda``, ``directml``, ``mps``, ``cpu``.
BACKEND_ENV_VAR = "HANZIGEN_BACKEND"

#: Set this environment variable to select the automatic mixed-precision dtype.
#: Accepted values: ``auto`` (default), ``fp16``, ``bf16``. bfloat16 has the same
#: exponent range as float32, so it is often the more stable choice on AMD
#: RDNA3/RDNA4 GPUs and never needs gradient scaling.
AMP_DTYPE_ENV_VAR = "HANZIGEN_AMP_DTYPE"

#: Substrings that ZLUDA reports as the device name. Used to spot ZLUDA because
#: its driver version is unparseable and it forwards the AMD marketing name.
_ZLUDA_DEVICE_MARKERS = (
    "radeon",
    "amd",
    "gfx",
    "navi",
)

#: Substrings that identify an AMD device name reported through ROCm/HIP.
_AMD_DEVICE_MARKERS = (
    "amd",
    "radeon",
    "instinct",
    "gfx",
)


class HardwareBackend(str, Enum):
    """
    Supported compute backends.
    """

    CUDA = "cuda"
    ROCM = "rocm"
    ZLUDA = "zluda"
    DIRECTML = "directml"
    MPS = "mps"
    CPU = "cpu"

    @property
    def is_gpu(self) -> bool:
        """
        Whether this backend runs on a GPU.
        """
        return self is not HardwareBackend.CPU

    @property
    def vendor(self) -> str:
        """
        Human readable vendor name.
        """
        match self:
            case HardwareBackend.CUDA:
                return "NVIDIA"
            case HardwareBackend.ROCM | HardwareBackend.ZLUDA | HardwareBackend.DIRECTML:
                return "AMD"
            case HardwareBackend.MPS:
                return "Apple"
            case _:
                return "CPU"

    @property
    def display_name(self) -> str:
        """
        Human readable backend name.
        """
        match self:
            case HardwareBackend.CUDA:
                return "CUDA"
            case HardwareBackend.ROCM:
                return "ROCm / HIP"
            case HardwareBackend.ZLUDA:
                return "ZLUDA (CUDA translation layer)"
            case HardwareBackend.DIRECTML:
                return "DirectML"
            case HardwareBackend.MPS:
                return "Metal Performance Shaders"
            case _:
                return "CPU"


@dataclass
class DeviceInfo:
    """
    Everything the project needs to know about the selected accelerator.
    """

    backend: HardwareBackend
    device: torch.device
    name: str = "CPU"
    #: Total memory of the accelerator in bytes, when it can be queried.
    total_memory: int | None = None
    #: Compute capability / GFX architecture, when it can be queried.
    compute_capability: str | None = None
    #: Whether the active PyTorch build is a HIP/ROCm build.
    is_rocm_build: bool = False
    #: Whether the active PyTorch build is an NVIDIA CUDA build.
    is_cuda_build: bool = False
    #: Extra free-form notes (e.g. why AMP was disabled).
    notes: list[str] = field(default_factory=list)

    @property
    def torch_device_type(self) -> str:
        """
        The ``device_type`` string that must be passed to ``torch.amp`` APIs.

        HIP devices are addressed as ``"cuda"`` by PyTorch, so this differs
        from :attr:`backend` for :class:`HardwareBackend.ROCM`.
        """
        return self.device.type

    @property
    def autocast_device_type(self) -> str:
        """
        The ``device_type`` accepted by :func:`torch.amp.autocast`.

        ``torch.amp.autocast`` only knows ``cuda``, ``cpu``, ``xpu``, ``hpu``,
        ``mps`` and ``mtia``. DirectML's ``"privateuseone"`` is not among them,
        so it is mapped to ``"cpu"``; autocast is reported as unsupported for
        DirectML anyway.
        """
        match self.device.type:
            case "cuda" | "cpu" | "xpu" | "hpu" | "mps" | "mtia":
                return self.device.type
            case _:
                return "cpu"


# --------------------------------------------------------------------------- #
# Backend detection
# --------------------------------------------------------------------------- #

_DETECTED: DeviceInfo | None = None
_DETECTION_LOGGED = False


def _env_backend_override() -> str | None:
    """
    Read the backend override from the environment.
    """
    raw = os.environ.get(BACKEND_ENV_VAR, "").strip().lower()
    if raw in ("", "auto"):
        return None
    return raw


def _clean_device_name(name: str) -> str:
    """
    Normalise a device name reported by a native backend.

    DirectML pads the name string with NUL bytes (it hands back a fixed-size
    wide-char buffer), which would otherwise leak into log output and any
    string comparison.
    """
    return name.replace("\x00", " ").strip()


def _cuda_device_name(index: int = 0) -> str:
    """
    Best-effort query of the CUDA-ish device name.
    """
    try:
        return _clean_device_name(str(torch.cuda.get_device_name(index)))
    except Exception:  # pragma: no cover - defensive, driver dependent
        return ""


def _cuda_device_properties(index: int = 0) -> Any | None:
    """
    Best-effort query of the CUDA-ish device properties.
    """
    try:
        return torch.cuda.get_device_properties(index)
    except Exception:  # pragma: no cover - defensive, driver dependent
        return None


def _cuda_compute_capability(index: int = 0) -> str | None:
    """
    Return the compute capability (or the GFX architecture) as a string.
    """
    properties = _cuda_device_properties(index)
    if properties is None:
        return None

    # ROCm builds attach `gcnArchName` (e.g. "gfx1201"); ZLUDA usually leaves it empty.
    gcn_arch = getattr(properties, "gcnArchName", None)
    if gcn_arch:
        return str(gcn_arch)

    major = getattr(properties, "major", None)
    minor = getattr(properties, "minor", None)
    if major is None or minor is None:
        return None

    # ZLUDA/ROCm report (0, 0) because there is no NVIDIA compute capability.
    if major == 0 and minor == 0:
        return None

    return f"{major}.{minor}"


def _looks_like_amd(name: str) -> bool:
    """
    Whether a device name looks like an AMD product.
    """
    lowered = name.lower()
    return any(marker in lowered for marker in _AMD_DEVICE_MARKERS)


def _detect_zluda(index: int = 0) -> bool:
    """
    Heuristically decide whether a CUDA-looking device is really ZLUDA.

    ZLUDA is loaded instead of the NVIDIA driver on an AMD card, and it uses
    the *official CUDA* PyTorch build, so ``torch.version.hip`` stays ``None``
    while the underlying device is AMD. The fingerprints are:

    * ``torch.cuda.get_device_name()`` returns the AMD marketing name;
    * ``torch.cuda.get_device_capability()`` reports ``(0, 0)``, because AMD
      hardware has no NVIDIA compute capability.

    A genuine ROCm/HIP build is already identified through
    ``torch.version.hip`` before this function runs, so an AMD device name
    combined with an unusable compute capability is a strong ZLUDA signal. The
    device name alone is *not* used as evidence, to avoid misclassifying an
    exotic driver.
    """
    try:
        capability = torch.cuda.get_device_capability(index)
    except Exception:  # pragma: no cover - defensive
        capability = (0, 0)

    properties = _cuda_device_properties(index)
    major = getattr(properties, "major", 0) if properties is not None else 0
    minor = getattr(properties, "minor", 0) if properties is not None else 0

    # NVIDIA compute capabilities start at 2.0; (0, 0) means "no SM version",
    # which only happens when the CUDA API is being emulated.
    if capability == (0, 0) or (major == 0 and minor == 0):
        return True

    return False


def _detect_rocm_masquerade(index: int = 0) -> bool:
    """
    Detect an AMD card reached through a CUDA-compatible layer while
    ``torch.version.hip`` is unset.

    Some ROCm-on-Windows distributions patch the CUDA namespace without setting
    ``torch.version.hip``. Those builds do report a real GFX architecture
    through ``gcnArchName``, which is a HIP-only property, so its presence is
    the discriminator between "ROCm in disguise" and "ZLUDA".
    """
    properties = _cuda_device_properties(index)
    if properties is None:
        return False

    gcn_arch = getattr(properties, "gcnArchName", None)
    if gcn_arch and str(gcn_arch).strip():
        return True

    return _looks_like_amd(_cuda_device_name(index)) and _cuda_compute_capability(index) is None


def configure_stdio() -> None:
    """
    Make standard output/error survive non-UTF-8 Windows consoles and appear
    in real time.

    Windows consoles default to a legacy code page (GBK/CP1252), where the
    diagnostic symbols used by this project raise ``UnicodeEncodeError`` and
    abort the run. Reconfiguring to UTF-8 with a replacement handler keeps the
    output readable instead of crashing.

    ``line_buffering=True`` is equally important: when the process is launched
    through a pipe (``run.py`` under PowerShell, or any ``python ... | tee``),
    CPython switches stdout to full block buffering, so ``print``/rich output is
    held in an 8 KiB buffer and training status (``Epoch Status``, ``Best model
    saved``) does not appear until the buffer fills or the process exits.
    Line buffering restores the normal "flush on every newline" behaviour so the
    per-epoch status shows up as it happens.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:  # pragma: no cover - platform dependent
            pass


def _import_directml(timeout_seconds: float = 20.0) -> Any | None:
    """
    Import ``torch_directml`` defensively.

    The import performs driver enumeration inside a native extension and can
    hang on machines without a DirectML capable adapter, so it runs in a
    watchdog thread and is abandoned after ``timeout_seconds``.
    """
    if platform.system() != "Windows":
        return None

    import importlib
    import threading

    result: dict[str, Any] = {}

    def _worker() -> None:
        try:
            result["module"] = importlib.import_module("torch_directml")
        except BaseException as error:  # pragma: no cover - driver dependent
            result["error"] = error

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():  # pragma: no cover - driver dependent
        return None

    return result.get("module")


def _build_device_info(backend: HardwareBackend, device: torch.device) -> DeviceInfo:
    """
    Collect details for a resolved backend/device pair.
    """
    info = DeviceInfo(
        backend=backend,
        device=device,
        is_rocm_build=torch.version.hip is not None,
        is_cuda_build=torch.version.cuda is not None and torch.version.hip is None,
    )

    index = device_index(device)

    if backend in (HardwareBackend.CUDA, HardwareBackend.ROCM, HardwareBackend.ZLUDA):
        info.name = _cuda_device_name(index) or backend.display_name
        info.compute_capability = _cuda_compute_capability(index)
        properties = _cuda_device_properties(index)
        if properties is not None:
            info.total_memory = getattr(properties, "total_memory", None)
    elif backend is HardwareBackend.MPS:
        info.name = "Apple Silicon GPU (MPS)"
    elif backend is HardwareBackend.DIRECTML:
        info.name = _directml_device_name(device) or "DirectML device"
    else:
        info.name = platform.processor() or platform.machine() or "CPU"

    if backend is HardwareBackend.ZLUDA:
        info.notes.append(
            "ZLUDA detected: PyTorch reports CUDA, but the device is an AMD GPU "
            "reached through a CUDA translation layer."
        )
    if backend is HardwareBackend.DIRECTML:
        info.notes.append(
            "DirectML does not support torch.amp autocast or GradScaler; mixed "
            "precision will be disabled and training runs in float32."
        )

    return info


def _directml_device_name(device: torch.device) -> str:
    """
    Query the human readable name of a DirectML device.
    """
    module = _import_directml()
    if module is None:
        return ""

    try:
        return _clean_device_name(str(module.device_name(int(device.index or 0))))
    except Exception:  # pragma: no cover - driver dependent
        return ""


def detect_backend(
    requested: str | None = None,
    index: int = 0,
) -> DeviceInfo:
    """
    Resolve the backend to use, honouring an explicit request first.

    Args:
        requested: A device string (``"cuda"``, ``"rocm"``, ``"zluda"``,
            ``"directml"``, ``"mps"``, ``"cpu"``, optionally with an index) or
            ``None``/``"auto"`` to auto-detect.
        index: Device ordinal used when probing device names.

    Returns:
        A populated :class:`DeviceInfo`.
    """
    device, forced_backend = _normalize_device(requested)

    if forced_backend is not None:
        return _build_device_info(forced_backend, device)

    # 1. Apple Silicon.
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return _build_device_info(HardwareBackend.MPS, torch.device("mps"))

    # 2. Any CUDA-looking accelerator: NVIDIA CUDA, AMD ROCm/HIP, or ZLUDA.
    if torch.cuda.is_available():
        if torch.version.hip is not None:
            backend = HardwareBackend.ROCM
        elif _detect_zluda(index):
            backend = HardwareBackend.ZLUDA
        elif _detect_rocm_masquerade(index):
            backend = HardwareBackend.ROCM
        else:
            backend = HardwareBackend.CUDA

        return _build_device_info(backend, torch.device("cuda", index))

    # 3. Windows DirectML.
    if _import_directml() is not None:
        return _build_device_info(HardwareBackend.DIRECTML, torch.device("privateuseone", 0))

    # 4. CPU fallback.
    return _build_device_info(HardwareBackend.CPU, torch.device("cpu"))


def _base_backend_for_type(device_type: str) -> HardwareBackend:
    """
    Map a bare torch device type to the backend that owns it.
    """
    match device_type:
        case "cpu":
            return HardwareBackend.CPU
        case "mps":
            return HardwareBackend.MPS
        case "privateuseone":
            return HardwareBackend.DIRECTML
        case "cuda":
            if torch.version.hip is not None:
                return HardwareBackend.ROCM
            return HardwareBackend.CUDA
        case _:
            return HardwareBackend.CPU


def reset_device_cache() -> None:
    """
    Discard the memoised auto-detection result.

    ``get_device_info(None)`` caches its answer because probing DirectML and
    querying driver properties is comparatively expensive and every model asks
    for it. Tests that simulate a different machine need to invalidate that
    cache, and production code that hot-plugs or re-configures a GPU can call
    this too.
    """
    global _DETECTED
    _DETECTED = None


def get_device_info(
    device: str | torch.device | None = None,
    refresh: bool = False,
) -> DeviceInfo:
    """
    Return (and memoise) the :class:`DeviceInfo` for ``device``.

    Auto-detection is cached because probing DirectML and querying driver
    properties is comparatively expensive and is repeated by every model.
    """
    global _DETECTED

    if device is None or (isinstance(device, str) and device.strip().lower() in ("", "auto")):
        override = _env_backend_override()
        if _DETECTED is not None and not refresh and override is None:
            return _DETECTED
        info = detect_backend(override)
        if override is None:
            _DETECTED = info
        return info

    resolved, forced_backend = _normalize_device(device)
    if forced_backend is not None:
        return _build_device_info(forced_backend, resolved)

    # An explicit device string: reuse the cached auto-detection when it points
    # at the same device type, otherwise rebuild from the device type itself.
    if not refresh:
        cached = get_device_info(None)
        if cached.device.type == resolved.type:
            return DeviceInfo(
                backend=cached.backend,
                device=resolved,
                name=cached.name,
                total_memory=cached.total_memory,
                compute_capability=cached.compute_capability,
                is_rocm_build=cached.is_rocm_build,
                is_cuda_build=cached.is_cuda_build,
                notes=list(cached.notes),
            )

    return _build_device_info(_base_backend_for_type(resolved.type), resolved)


# --------------------------------------------------------------------------- #
# Device string normalisation
# --------------------------------------------------------------------------- #

#: Aliases accepted on the command line and mapped to a canonical torch device.
_BACKEND_ALIASES: dict[str, HardwareBackend] = {
    "cuda": HardwareBackend.CUDA,
    "nvidia": HardwareBackend.CUDA,
    "rocm": HardwareBackend.ROCM,
    "hip": HardwareBackend.ROCM,
    "amd": HardwareBackend.ROCM,
    "zluda": HardwareBackend.ZLUDA,
    "directml": HardwareBackend.DIRECTML,
    "dml": HardwareBackend.DIRECTML,
    "mps": HardwareBackend.MPS,
    "metal": HardwareBackend.MPS,
    "cpu": HardwareBackend.CPU,
}


def _normalize_device(
    device: str | torch.device | None,
) -> tuple[torch.device, HardwareBackend | None]:
    """
    Turn a user supplied device string into a torch device plus an optional
    forced backend.

    Returns:
        ``(torch_device, forced_backend)``. ``forced_backend`` is ``None`` when
        the value should be auto-detected.

    A ``torch.device`` argument never forces a backend: ``cuda:0`` is the same
    object on NVIDIA, AMD ROCm and ZLUDA, so the vendor has to be detected.
    Only an explicit *string* names a backend.

    Examples:
        >>> _normalize_device(None)
        (device(type='cuda', index=0), None)   # on a CUDA machine
        >>> _normalize_device("rocm")
        (device(type='cuda', index=0), <HardwareBackend.ROCM: 'rocm'>)
        >>> _normalize_device("directml:0")
        (device(type='privateuseone', index=0), <HardwareBackend.DIRECTML: 'directml'>)
        >>> _normalize_device(torch.device("cuda:0"))
        (device(type='cuda', index=0), None)   # vendor determined by detection
    """
    if device is None:
        return _default_device_for(None), None

    if isinstance(device, torch.device):
        # A bare torch.device cannot say *which* vendor it is: on a ROCm build
        # HIP also uses the "cuda" namespace, so torch.device("cuda:0") is
        # equally valid on NVIDIA, AMD ROCm and ZLUDA. Report no forced backend
        # and let detection decide, rather than assuming NVIDIA.
        return device, None

    raw = str(device).strip().lower()
    if raw in ("", "auto"):
        return _default_device_for(None), None

    name, _, index_part = raw.partition(":")
    index = int(index_part) if index_part.strip().isdigit() else 0

    backend = _BACKEND_ALIASES.get(name)
    if backend is None:
        # Not a known alias: hand the raw string to PyTorch so that exotic
        # device strings still work.
        return torch.device(device), None

    return _default_device_for(backend, index), backend


def _torch_type_for(backend: HardwareBackend) -> str:
    """
    The torch device type that carries a given backend.
    """
    match backend:
        case HardwareBackend.CUDA | HardwareBackend.ROCM | HardwareBackend.ZLUDA:
            return "cuda"
        case HardwareBackend.DIRECTML:
            return "privateuseone"
        case HardwareBackend.MPS:
            return "mps"
        case _:
            return "cpu"


def _default_device_for(
    backend: HardwareBackend | None,
    index: int | None = None,
) -> torch.device:
    """
    Build the torch device for a backend, or auto-detect when ``backend`` is None.
    """
    if backend is None:
        if torch.cuda.is_available():
            return torch.device("cuda", index or 0)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        if _import_directml() is not None:
            return torch.device("privateuseone", index or 0)
        return torch.device("cpu")

    device_type = _torch_type_for(backend)
    if device_type == "cpu":
        return torch.device("cpu")

    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"[error] Device '{backend.value}' was requested but "
            "torch.cuda.is_available() "
            "is False. Install a CUDA/ROCm/ZLUDA capable PyTorch build, or pass "
            "--device cpu. See docs/AMD_SETUP.md."
        )

    if device_type == "privateuseone" and _import_directml() is None:
        raise RuntimeError(
            "[error] Device 'directml' was requested but torch_directml could not be "
            "imported. Install it with 'uv pip install torch-directml', or pass "
            "--device cpu. See docs/AMD_SETUP.md."
        )

    return torch.device(device_type, 0 if index is None else index)


def normalize_device(
    device: str | torch.device | None,
) -> tuple[torch.device, HardwareBackend | None]:
    """
    Public alias of :func:`_normalize_device`.
    """
    return _normalize_device(device)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def device_index(device: torch.device) -> int:
    """
    Return the ordinal of ``device``, defaulting to ``0``.
    """
    return int(device.index) if device.index is not None else 0


def is_accelerator(device: torch.device | str | None) -> bool:
    """
    Whether a device is a GPU-like accelerator.

    Used to decide whether pinned host memory is worth requesting.
    """
    if device is None:
        return False
    if isinstance(device, str):
        return device.strip().lower() not in ("", "cpu", "auto")
    return device.type != "cpu"


def is_amd(device: torch.device | str | None = None) -> bool:
    """
    Whether the selected backend drives an AMD GPU.
    """
    return get_device_info(device).backend.vendor == "AMD"


def describe_device(device: torch.device | str | None = None) -> str:
    """
    One-line, human readable description of the active device.
    """
    info = get_device_info(device)
    if info.backend is HardwareBackend.CPU:
        return "CPU"

    details = [f"{info.backend.display_name}: {info.name}"]
    if info.device.index:
        details.append(f"index={info.device.index}")
    if info.compute_capability:
        details.append(f"arch={info.compute_capability}")
    if info.total_memory:
        details.append(f"vram={info.total_memory / 1024**3:.2f} GB")

    return ", ".join(details)


# --------------------------------------------------------------------------- #
# Automatic mixed precision
# --------------------------------------------------------------------------- #


def _autocast_device_type(device: torch.device | str | None) -> str:
    """
    The ``device_type`` string accepted by :func:`torch.amp.autocast` for a device.
    """
    return get_device_info(device).autocast_device_type


def no_autocast(device: torch.device | str | None):
    """
    Context manager that explicitly disables autocast for the given device.

    Useful for numerically sensitive regions (the VQ codebook lookup, for
    instance) that must stay in float32 even while the surrounding training
    loop runs in mixed precision. Safe on backends without autocast support.
    """
    device_type = _autocast_device_type(device)
    try:
        return torch.amp.autocast(device_type=device_type, enabled=False)
    except (TypeError, RuntimeError, ValueError):  # pragma: no cover - version dependent
        try:
            return torch.autocast(device_type=device_type, enabled=False)
        except Exception:
            return nullcontext()


def supports_amp(device: torch.device | str | None = None) -> bool:
    """
    Whether ``torch.amp.autocast`` is usable on this backend.

    DirectML is the only supported backend without autocast support, because
    ``torch.amp`` has no ``privateuseone`` device type.
    """
    info = get_device_info(device)
    return info.backend is not HardwareBackend.DIRECTML and info.backend is not HardwareBackend.CPU


def supports_grad_scaler(device: torch.device | str | None = None) -> bool:
    """
    Whether :class:`torch.amp.GradScaler` is usable on this backend.

    Gradient scaling exists to compensate for the narrow exponent range of
    IEEE float16, which only matters on CUDA-like devices. DirectML does not
    support it.
    """
    info = get_device_info(device)
    return info.backend in (
        HardwareBackend.CUDA,
        HardwareBackend.ROCM,
        HardwareBackend.ZLUDA,
    )


def supports_bfloat16(device: torch.device | str | None = None) -> bool:
    """
    Best-effort check for native bfloat16 support on ``device``.

    RDNA3/RDNA4 AMD GPUs (gfx1100/gfx1200) and most recent NVIDIA cards support
    bfloat16 natively. The check is used for diagnostics and to warn when a user
    requests bf16 that a device cannot accelerate; it never raises.
    """
    try:
        info = get_device_info(device)
        if info.device.type == "cuda":
            return bool(torch.cuda.is_bf16_supported())
        if info.device.type == "cpu":
            return True
    except Exception:  # pragma: no cover - build dependent
        return False
    return False


def _parse_amp_dtype_override() -> str:
    """
    Read the AMP dtype override from the environment.

    Returns ``"auto"``, ``"fp16"`` or ``"bf16"``. An unrecognised value prints a
    warning and falls back to ``"auto"``.
    """
    raw = os.environ.get(AMP_DTYPE_ENV_VAR, "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "default": "auto",
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
    }
    if raw not in aliases:
        print(
            f"[warn] Unrecognised {AMP_DTYPE_ENV_VAR}='{raw}', falling back to "
            "'auto' (valid values: auto, fp16, bf16)."
        )
        return "auto"
    return aliases[raw]


def resolve_amp_dtype(device: torch.device | str | None = None) -> torch.dtype:
    """
    Resolve the autocast dtype to use for ``device``.

    Defaults preserve the original behaviour of this project (``float16`` on
    GPU, ``bfloat16`` on CPU). Set ``HANZIGEN_AMP_DTYPE=bf16`` to switch the GPU
    autocast to bfloat16, which has the same exponent range as float32 and is
    therefore more numerically forgiving on AMD RDNA3/RDNA4 hardware -- and
    needs no gradient scaling.
    """
    override = _parse_amp_dtype_override()

    if override == "fp16":
        return torch.float16
    if override == "bf16":
        info = get_device_info(device)
        if info.device.type != "cpu" and not supports_bfloat16(device):
            print(
                f"[warn] {AMP_DTYPE_ENV_VAR}=bf16 was requested but "
                f"'{info.device}' does not report bfloat16 support; continuing "
                "anyway."
            )
        return torch.bfloat16

    # Automatic selection: bfloat16 on CPU (strictly faster there, no precision
    # cost), float16 on every accelerator (the established default).
    info = get_device_info(device)
    return torch.bfloat16 if info.device.type == "cpu" else torch.float16


def amp_requires_grad_scaler(device: torch.device | str | None = None) -> bool:
    """
    Whether the resolved AMP dtype needs gradient scaling.

    bfloat16 has the same exponent range as float32, so it needs no
    ``GradScaler``. Returning False lets :func:`make_grad_scaler` hand back a
    no-op scaler and keeps a single code path for both dtypes.
    """
    return resolve_amp_dtype(device) == torch.float16


def autocast_dtype(device: torch.device | str | None = None) -> torch.dtype:
    """
    The mixed precision dtype that is safe on this backend.

    float16 is the default on CUDA-like devices (matching the project's
    original behaviour). bfloat16 is often the better choice on AMD RDNA3/RDNA4
    GPUs and is selected with ``HANZIGEN_AMP_DTYPE=bf16``; see
    :func:`resolve_amp_dtype`.
    """
    return resolve_amp_dtype(device)


def autocast_context(
    device: torch.device | str | None = None,
    enabled: bool = True,
):
    """
    Build an autocast context manager for ``device``.

    Returns :data:`contextlib.nullcontext` whenever mixed precision is not
    supported or not requested, so callers can use it unconditionally.
    """
    if not enabled or not supports_amp(device):
        return nullcontext()

    info = get_device_info(device)
    device_type = info.autocast_device_type

    try:
        return torch.amp.autocast(
            device_type=device_type,
            dtype=autocast_dtype(device),
            enabled=True,
        )
    except (TypeError, RuntimeError, ValueError):  # pragma: no cover - version dependent
        try:
            return torch.autocast(device_type=device_type, dtype=autocast_dtype(device))
        except Exception:
            return nullcontext()


class _GradScalerFallback:
    """
    A no-op stand-in for :class:`torch.amp.GradScaler`.

    Some backends (notably DirectML) cannot scale gradients. Rather than
    branching in every training loop, callers get this object, which keeps the
    usual ``scale``/``unscale_``/``step``/``update`` call sequence valid while
    performing plain, unscaled optimisation steps.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = False
        self._scale = 1.0

    def is_enabled(self) -> bool:
        """
        Mirror :meth:`torch.amp.GradScaler.is_enabled`.
        """
        return False

    def scale(self, outputs):
        return outputs

    def unscale_(self, optimizer) -> None:
        return None

    def step(self, optimizer, *args, **kwargs):
        return optimizer.step(*args, **kwargs)

    def update(self, new_scale=None) -> None:
        return None

    def get_scale(self) -> float:
        return self._scale

    def state_dict(self) -> dict[str, Any]:
        return {"enabled": False, "scale": self._scale}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        return None


def scaler_is_enabled(scaler: Any) -> bool:
    """
    Whether a gradient scaler will actually scale.

    ``torch.amp.GradScaler`` exposes ``is_enabled()`` and has **no** ``enabled``
    attribute, so a naive ``getattr(scaler, "enabled", False)`` always reports
    ``False`` for a real scaler. Reading it that way silently disabled autocast
    even when the user explicitly asked for mixed precision.

    This helper understands a real scaler, the no-op fallback, and ``None``.
    """
    if scaler is None:
        return False

    is_enabled = getattr(scaler, "is_enabled", None)
    if callable(is_enabled):
        try:
            return bool(is_enabled())
        except Exception:  # pragma: no cover - defensive
            return False

    return bool(getattr(scaler, "enabled", False))


def make_grad_scaler(
    device: torch.device | str | None = None,
    enabled: bool = True,
) -> Any:
    """
    Build a gradient scaler that is appropriate for ``device``.

    On CUDA/ROCm/ZLUDA with a float16 AMP dtype this is a real
    :class:`torch.amp.GradScaler`. A no-op fallback is returned when the dtype
    is bfloat16 (which needs no scaling), when the backend cannot scale, or when
    AMP is disabled, so mixed precision requests degrade gracefully instead of
    raising.
    """
    if not enabled:
        return _GradScalerFallback()

    if not supports_grad_scaler(device):
        return _GradScalerFallback()

    if not amp_requires_grad_scaler(device):
        # bfloat16 has float32's exponent range; gradients need no scaling.
        return _GradScalerFallback()

    info = get_device_info(device)
    try:
        # PyTorch >= 2.3 accepts `device`; the older positional form is `device="cuda"`.
        return torch.amp.GradScaler(device=info.autocast_device_type, enabled=True)
    except TypeError:  # pragma: no cover - older PyTorch
        try:
            return torch.amp.GradScaler(info.autocast_device_type, enabled=True)
        except Exception:
            return _GradScalerFallback()
    except Exception:  # pragma: no cover - defensive
        return _GradScalerFallback()
