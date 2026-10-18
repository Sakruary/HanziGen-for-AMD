"""
Backend-aware entry point for HanziGen.

Every ``scripts/bat/*.bat`` and ``scripts/sh/*.sh`` wrapper calls this module so
that one place decides which compute backend to use. It exists because the
wrapper scripts cannot inspect PyTorch themselves (and because batch/bash
backend detection is hard to get right).

Usage:
    uv run python run.py --backend auto  train_vqvae.py --num_epochs 1
    uv run python run.py --backend rocm  train_ldm.py
    uv run python run.py --backend directml inference.py
    uv run python run.py --list

The ``--backend`` value is resolved in this order:

1. the explicit ``--backend`` argument;
2. the ``HANZIGEN_BACKEND`` environment variable;
3. the presence of a single ``constraints/amd-*.txt`` marker that says the user
   already chose an AMD path;
4. whatever ``torch`` itself reports (and ``auto`` when torch is not installed).

The resolved backend is exported as ``HANZIGEN_BACKEND`` so that
:func:`utils.hardware.hardware_utils.select_device` agrees with the launcher,
and is passed to the target script as ``--device`` unless the script already
sets it.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CONSTRAINTS_DIR = PROJECT_ROOT / "constraints"
BACKEND_ENV_VAR = "HANZIGEN_BACKEND"

#: Constraints file to backend value.
CONSTRAINT_MARKERS = {
    "amd-rocm.txt": "rocm",
    "amd-zluda.txt": "zluda",
    "amd-directml.txt": "directml",
    "cpu.txt": "cpu",
    "nvidia-cuda.txt": "cuda",
}

#: AMD's TheRock index, which is the only source of Windows ROCm wheels for
#: RDNA4. Verified working on Windows 11 + RX 9070 XT (gfx1201) with
#: torch 2.10.0+rocm7.13.0a20260511. The download.pytorch.org/whl/rocm<ver>
#: indexes contain Linux wheels only, so they are not offered here.
THEROCK_INDEX_URL = "https://rocm.nightlies.amd.com/v2/gfx120X-all/"

#: Linux ROCm releases that publish wheels on download.pytorch.org.
ROCM_INDEX_TAGS = ("6.3", "6.4", "7.0", "7.1", "7.2")

#: Overrides for the automatic ROCm install. All optional.
ROCM_WHEEL_INDEX_ENV_VAR = "HANZIGEN_ROCM_WHEEL_INDEX"
TORCH_VERSION_ENV_VAR = "HANZIGEN_TORCH_VERSION"

#: The ROCm build the launcher installs when it has to bootstrap from scratch.
#: A ROCm wheel is identified by its `+rocm<release><datestamp>` local version,
#: so this must be the full string -- a bare `torch==2.10.0` can be satisfied
#: by a CPU wheel, and PyPI's higher version numbers win over an
#: --extra-index-url if it is present.
DEFAULT_ROCM_TORCH_VERSION = "2.10.0+rocm7.13.0a20260511"
DEFAULT_ROCM_TORCHVISION_VERSION = "0.25.0+rocm7.13.0a20260511"
DEFAULT_ROCM_TORCHAUDIO_VERSION = "2.10.0+rocm7.13.0a20260511"
DEFAULT_ROCM_SDK_VERSION = "7.13.0a20260511"

#: Set to 1 to forbid the launcher from ever running `pip install`.
NO_INSTALL_ENV_VAR = "HANZIGEN_NO_PIP_INSTALL"


@dataclass
class Resolution:
    """
    The outcome of backend resolution.
    """

    backend: str
    reason: str
    torch_device_type: str | None = None

    def describe(self) -> str:
        detail = f"[backend] {self.backend}"
        if self.torch_device_type:
            detail += f" (torch device '{self.torch_device_type}')"
        return f"{detail} - {self.reason}"


#: Values accepted by ``--backend``. Validated here rather than by argparse so
#: that an empty value can be normalised to "auto" instead of being rejected --
#: the wrapper scripts leave DEVICE empty by default.
KNOWN_BACKENDS = (
    "auto",
    "cuda",
    "rocm",
    "hip",
    "amd",
    "zluda",
    "directml",
    "dml",
    "mps",
    "cpu",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Resolve the compute backend, then run a HanziGen script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python run.py --backend rocm train_vqvae.py --num_epochs 100\n"
            "  uv run python run.py report\n"
            "  uv run python run.py --list\n"
        ),
    )
    parser.add_argument(
        "script",
        nargs="?",
        help="Script to run, e.g. train_vqvae.py (or `report` for check_hardware.py)",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the target script",
    )
    parser.add_argument(
        "--backend",
        default=None,
        metavar="NAME",
        help=(
            "Compute backend to use. One of: "
            + ", ".join(KNOWN_BACKENDS)
            + " (default: auto). An empty value also means auto."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only print the backend resolution result and exit",
    )
    return parser


def _probe_torch() -> tuple[bool, str | None, bool, bool]:
    """
    Return ``(torch_available, hip_version, cuda_available, mps_available)``.
    """
    try:
        import torch
    except ImportError:
        return False, None, False, False

    try:
        mps = bool(torch.backends.mps.is_available() and torch.backends.mps.is_built())
    except Exception:  # pragma: no cover - build dependent
        mps = False

    return True, torch.version.hip, bool(torch.cuda.is_available()), mps


def _probe_directml() -> bool:
    """
    Whether ``torch_directml`` is importable.
    """
    try:
        import torch_directml  # noqa: F401
    except Exception:
        return False
    return True


def resolve_backend(requested: str | None) -> Resolution:
    """
    Decide the backend to use and explain why.
    """
    requested = (requested or "").strip().lower() or None

    if requested and requested != "auto":
        return Resolution(requested, "explicit --backend", None)

    environment = os.environ.get(BACKEND_ENV_VAR, "").strip().lower()
    if environment and environment != "auto":
        return Resolution(environment, f"${BACKEND_ENV_VAR} environment variable", None)
    present = [name for name in CONSTRAINT_MARKERS if (CONSTRAINTS_DIR / name).exists()]
    if len(present) == 1:
        return Resolution(
            CONSTRAINT_MARKERS[present[0]],
            f"constraints/{present[0]} is the only backend marker present",
            None,
        )

    torch_available, hip_version, cuda_available, mps_available = _probe_torch()

    if not torch_available:
        return Resolution("auto", "PyTorch is not installed; torch will decide", None)

    if cuda_available:
        if hip_version is not None:
            return Resolution("rocm", f"torch.version.hip={hip_version}", "cuda")
        import torch

        return Resolution(
            "cuda",
            f"torch.cuda is available (torch.version.cuda={torch.version.cuda})",
            "cuda",
        )

    if mps_available:
        return Resolution("mps", "torch.backends.mps is available", "mps")

    if _probe_directml():
        return Resolution("directml", "torch_directml is installed", "privateuseone")

    return Resolution("cpu", "no accelerator detected", "cpu")


def _run_cuda(initial: str | None) -> tuple[str, str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "❌ CUDA was requested but torch.cuda.is_available() is False.\n"
            "   Install a CUDA build of PyTorch:\n"
            "     uv pip install -r constraints/nvidia-cuda.txt"
        )
    if torch.version.hip is not None:
        print(
            "   note: this PyTorch is a ROCm/HIP build, so 'cuda' resolves to "
            "ROCm. Use --backend rocm to silence this note."
        )
    return initial or "cuda", "torch.cuda is available"


def _run_rocm(initial: str | None) -> tuple[str, str]:
    import torch

    if torch.version.hip is not None and torch.cuda.is_available():
        return initial or "rocm", f"ROCm/HIP {torch.version.hip} is active"

    if torch.version.hip is None:
        # A successful install re-execs this process, so reaching the next line
        # means nothing was installed. Reloading torch in place is not an
        # option: a CUDA/CPU build is already imported and reloading a native
        # extension fails. Report the situation instead.
        _try_install_rocm()
        raise RuntimeError(
            "❌ ROCm was requested but no HIP device is visible.\n"
            "   * Check whether PyTorch is a ROCm build:\n"
            '     python -c "import torch; print(torch.version.hip)"\n'
            "     A ROCm build prints a version such as 7.13.26176; None means\n"
            "     you have a CPU or NVIDIA build.\n"
            "   * Linux / WSL2: check the driver with 'rocminfo | grep gfx'. An\n"
            "     RX 9070 XT (gfx1201) should be listed.\n"
            "   * Windows: follow the ROCm-on-Windows section of docs/AMD_SETUP.md.\n"
            "   * Install a ROCm build of PyTorch, then re-run:\n"
            "     uv pip install -r constraints/amd-rocm.txt"
        )

    raise RuntimeError(
        "❌ ROCm was requested, this PyTorch is a HIP build, but no HIP device is\n"
        "   visible to it. Check the driver:\n"
        "   * Linux / WSL2: 'rocminfo | grep gfx' should list your GPU.\n"
        "   * Windows: follow docs/AMD_SETUP.md, section 'route A'."
    )


def _try_install_rocm() -> None:
    """
    Best-effort ``uv pip install`` of a ROCm PyTorch build.

    Only attempted when PyTorch is a CPU/CUDA build but ROCm was requested, and
    skipped entirely when ``HANZIGEN_NO_PIP_INSTALL=1``.

    The sequence below is the one verified to work on Windows 11 + RX 9070 XT.
    Three details matter, each of which was got wrong at least once:

    1. **No ``--extra-index-url https://pypi.org/simple``.** PyPI carries a
       higher version number than the ROCm nightlies, so it wins and you end up
       with a ``+cpu`` build. Dependencies are installed from PyPI in a separate
       step instead.
    2. **Exact local version pins.** ROCm wheels are identified by
       ``+rocm<release><datestamp>``; a bare ``torch==2.10.0`` can be satisfied
       by a CPU wheel.
    3. **The ROCm SDK packages are required.** ``import torch`` on Windows
       imports ``rocm_sdk``, which ships in the ``rocm`` meta package (an
       sdist). Without it the import fails outright.

    Overridable with ``HANZIGEN_ROCM_WHEEL_INDEX`` and
    ``HANZIGEN_TORCH_VERSION``.
    """
    import subprocess
    import urllib.request

    if os.environ.get(NO_INSTALL_ENV_VAR) == "1":
        print(f"   {NO_INSTALL_ENV_VAR}=1 is set; not installing anything")
        return

    is_windows = os.name == "nt"
    explicit_index = os.environ.get(ROCM_WHEEL_INDEX_ENV_VAR, "").strip()

    if explicit_index:
        candidates = [explicit_index]
    elif is_windows:
        # download.pytorch.org publishes Linux ROCm wheels only.
        candidates = [THEROCK_INDEX_URL]
    else:
        candidates = [
            f"https://download.pytorch.org/whl/rocm{tag}" for tag in ROCM_INDEX_TAGS
        ]

    print("   PyTorch is not a ROCm build; looking for a usable ROCm index ...")

    reachable = [url for url in candidates if _index_is_reachable(url, urllib)]
    if not reachable:
        print(
            "   no reachable ROCm index found.\n"
            "   * Windows: set HANZIGEN_ROCM_WHEEL_INDEX to AMD's TheRock index:\n"
            f"     {THEROCK_INDEX_URL}\n"
            "   * Linux: check https://download.pytorch.org/whl/ for the\n"
            "     rocm<version> directory that matches your driver."
        )
        return

    index_url = reachable[-1]
    torch_version = os.environ.get(TORCH_VERSION_ENV_VAR, "").strip()

    print(f"   installing ROCm PyTorch from {index_url} ...")

    base = ["uv", "pip", "install", "--index-url", index_url]
    if "nightlies" in index_url or "nightly" in index_url:
        base.append("--prerelease=allow")

    if torch_version:
        # User asked for a specific version: let the resolver pull the matching
        # companions, but still keep PyPI out of it.
        steps = [base + [f"torch=={torch_version}"]]
    else:
        # Dependencies first, from PyPI, so that it does not need to be a
        # fallback index during the ROCm step.
        steps = [
            [
                "uv", "pip", "install",
                "--index-url", "https://pypi.org/simple",
                "filelock", "fsspec", "jinja2", "networkx", "sympy",
                "numpy", "pillow", "typing-extensions", "setuptools",
            ],
            # Then the exact ROCm trio plus the SDK, with --no-deps so nothing
            # can substitute a CPU wheel.
            base + [
                "--no-deps",
                f"torch=={DEFAULT_ROCM_TORCH_VERSION}",
                f"torchvision=={DEFAULT_ROCM_TORCHVISION_VERSION}",
                f"torchaudio=={DEFAULT_ROCM_TORCHAUDIO_VERSION}",
                f"rocm-sdk-core=={DEFAULT_ROCM_SDK_VERSION}",
                f"rocm-sdk-libraries-gfx120x-all=={DEFAULT_ROCM_SDK_VERSION}",
                f"rocm=={DEFAULT_ROCM_SDK_VERSION}",
            ],
        ]

    for step in steps:
        try:
            subprocess.run(step, check=True)
        except FileNotFoundError:
            print("   'uv' was not found on PATH; install PyTorch manually, then re-run")
            return
        except subprocess.CalledProcessError as error:
            print(f"   install failed (exit {error.returncode}); continuing as-is")
            return
        except Exception as error:  # pragma: no cover - defensive
            print(f"   install failed ({error}); continuing as-is")
            return

    # A fresh interpreter is the only reliable way to pick up the new build.
    _reexec()


def _index_is_reachable(index_url: str, urllib_module) -> bool:
    """
    Whether a package index answers over HTTP.
    """
    for suffix in ("/torch/", "/"):
        try:
            probe_url = index_url.rstrip("/") + suffix
            with urllib_module.request.urlopen(probe_url, timeout=10) as response:
                if response.status == 200:
                    return True
        except Exception:
            continue
    return False


def _reexec() -> None:
    """
    Re-run this process so the newly installed PyTorch is imported.
    """
    print("   re-launching with the new PyTorch build ...")
    os.execv(sys.executable, [sys.executable, *sys.argv])


def _run_zluda(initial: str | None) -> tuple[str, str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "❌ ZLUDA was requested but torch.cuda.is_available() is False.\n"
            "   ZLUDA replaces the CUDA runtime DLLs, so PyTorch must be a CUDA\n"
            "   build and the ZLUDA DLLs must sit in site-packages/torch/lib.\n"
            "   See docs/AMD_SETUP.md, section 'route B'."
        )
    name = ""
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - driver dependent
        pass
    return initial or "zluda", f"CUDA namespace is usable; device reports '{name}'"


def _run_directml(initial: str | None) -> tuple[str, str]:
    try:
        import torch_directml
    except Exception as error:
        raise RuntimeError(
            "❌ DirectML was requested but torch_directml is not installed.\n"
            "   uv pip install -r constraints/amd-directml.txt\n"
            f"   (import failed with: {error})"
        ) from error

    device_count = getattr(torch_directml, "device_count", lambda: 1)()
    if device_count <= 0:
        raise RuntimeError("❌ torch_directml reports no DirectML capable adapter")

    index = 0
    if initial and ":" in initial:
        try:
            index = int(initial.split(":", 1)[1])
        except ValueError:
            index = 0

    # Verify the device actually materialises before handing it to the script.
    for candidate in (index, 0):
        try:
            torch_directml.device(candidate)
            break
        except Exception:
            continue
    else:  # pragma: no cover - driver dependent
        raise RuntimeError("❌ could not create a torch_directml device")

    return f"directml:{index}", f"{device_count} DirectML adapter(s) available"


def _run_cpu(initial: str | None) -> tuple[str, str]:
    return initial or "cpu", "CPU selected"


def _resolve_device_argument(backend: str) -> tuple[str, str]:
    """
    Turn a backend name into the ``--device`` value to forward.
    """
    match backend:
        case "cuda":
            return _run_cuda(None)
        case "rocm" | "hip" | "amd":
            return _run_rocm(None)
        case "zluda":
            return _run_zluda(None)
        case "directml" | "dml":
            return _run_directml(None)
        case "mps":
            return "mps", "MPS selected"
        case "cpu":
            return _run_cpu(None)
        case _:
            # "auto" (and anything unrecognised) lets torch decide inside the
            # target script. Nothing is probed here, so this also stays cheap
            # for the CPU-only data-preparation scripts.
            return "auto", "auto-detection deferred to PyTorch"


def _normalize_backend_value(value: str | None) -> str | None:
    """
    Treat an empty or whitespace ``--backend`` value as "not specified".

    The wrapper scripts assign ``DEVICE=""`` by default so that
    ``--backend "%DEVICE%"`` becomes ``--backend ""`` on Windows and
    ``--backend ""`` in bash. Both must mean "auto-detect", not "invalid".
    """
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def main() -> int:
    from utils.hardware.hardware_utils import configure_stdio

    configure_stdio()

    parser = build_parser()
    namespace = parser.parse_args()

    if not namespace.script:
        parser.print_help()
        return 2

    # `report` is a convenience alias for the hardware diagnostic.
    list_only = namespace.list
    if namespace.script == "report":
        namespace.script = "check_hardware.py"
        list_only = False
    elif namespace.script == "check_hardware.py":
        # Its own --list does not exist; make `run.py --list check_hardware.py`
        # mean the same as running the report.
        list_only = False

    backend_value = _normalize_backend_value(namespace.backend)
    if backend_value is not None and backend_value not in KNOWN_BACKENDS:
        print(
            f"❌ unknown backend '{namespace.backend}'\n"
            f"   valid values: {', '.join(KNOWN_BACKENDS)}",
            file=sys.stderr,
        )
        return 2

    resolution = resolve_backend(backend_value)

    try:
        device, detail = _resolve_device_argument(resolution.backend)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1

    print(f"[backend] {resolution.backend}: {detail}")
    print(f"[backend] resolved because: {resolution.reason}")
    print(f"[device]  --device {device}")

    if list_only:
        return 0

    script_path = PROJECT_ROOT / namespace.script
    if not script_path.exists():
        print(f"❌ script not found: {script_path}", file=sys.stderr)
        return 2

    os.environ[BACKEND_ENV_VAR] = resolution.backend

    forwarded = list(namespace.args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if "--list" in forwarded and namespace.script == "check_hardware.py":
        # `report`/`check_hardware.py` has no --list of its own.
        forwarded = [item for item in forwarded if item != "--list"]
    # `--device` and `--device=cpu` are both valid spellings.
    if not any(item == "--device" or item.startswith("--device=") for item in forwarded):
        forwarded = ["--device", device, *forwarded]

    # Make the project importable without shadowing anything the caller put on
    # PYTHONPATH ahead of it (which is how configs/ is overridden).
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    sys.argv = [str(script_path), *forwarded]

    print(f"[run]     {namespace.script} {' '.join(forwarded)}")
    print("-" * 72)

    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as exit_error:
        return int(exit_error.code or 0)

    return 0


if __name__ == "__main__":
    # ROCm on Windows can hang during interpreter shutdown (HIP context
    # teardown). By the time main() returns, every checkpoint has been saved,
    # TensorBoard logs closed, and console output flushed (stdout is line
    # buffered via configure_stdio), so hard-exit to skip the teardown hang.
    # On non-Windows / non-ROCm this is still a correct, slightly eager exit.
    os._exit(main())
