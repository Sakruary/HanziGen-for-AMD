"""
Hardware diagnostic for HanziGen.

Reports which compute backend is active, why, and whether the operations this
project actually relies on can run on it. Run this first whenever a machine
misbehaves -- it is the fastest way to tell an NVIDIA/CUDA box apart from an
AMD/ROCm, AMD/ZLUDA, DirectML or CPU box.

Usage:
    uv run python check_hardware.py
    uv run python check_hardware.py --device rocm
    uv run python check_hardware.py --benchmark
    uv run python check_hardware.py --installed-ops
"""

from __future__ import annotations

import argparse
import platform
import sys
import time

import torch

from utils.hardware.hardware_utils import (
    BACKEND_ENV_VAR,
    HardwareBackend,
    configure_stdio,
    describe_device,
    get_device_info,
    resolve_amp_settings,
    select_device,
    supports_amp,
    supports_grad_scaler,
)

configure_stdio()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the AMD/CUDA/CPU compute setup used by HanziGen",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to diagnose: auto (default), cuda, rocm, zluda, directml, mps, cpu",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a short throughput benchmark (conv2d + matmul + autograd step)",
    )
    parser.add_argument(
        "--installed-ops",
        action="store_true",
        help="List every op registered for the active device type (very verbose)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if any functional check fails",
    )
    return parser.parse_args()


def print_environment() -> None:
    print("=" * 72)
    print("Environment")
    print("=" * 72)
    print(f"  OS                 : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"  Machine            : {platform.machine()}")
    print(f"  Python             : {sys.version.split()[0]} ({sys.executable})")
    print(f"  PyTorch            : {torch.__version__}")
    print(f"  torch.version.cuda : {torch.version.cuda}")
    print(f"  torch.version.hip  : {torch.version.hip}")
    print(f"  CUDA available     : {torch.cuda.is_available()}")
    if torch.version.hip is not None:
        print("  Build type         : ROCm / HIP build (AMD)")
    elif torch.version.cuda is not None:
        print("  Build type         : CUDA build (NVIDIA, or ZLUDA if an AMD GPU is present)")
    else:
        # torch-directml ships a build that reports neither, so mention it
        # explicitly to avoid the reading "this is a CPU-only wheel".
        try:
            import torch_directml  # noqa: F401

            print("  Build type         : CPU/CUDA-less torch, driven through torch-directml")
        except Exception:
            print("  Build type         : CPU build")
    print(f"  MPS available      : {torch.backends.mps.is_available()}")
    print(f"  {BACKEND_ENV_VAR:<18} : {__import__('os').environ.get(BACKEND_ENV_VAR, '(unset)')}")


def print_selected_device(device: torch.device) -> None:
    info = get_device_info(device)

    print()
    print("=" * 72)
    print("Selected backend")
    print("=" * 72)
    print(f"  Backend            : {info.backend.display_name} ({info.backend.value})")
    print(f"  Vendor             : {info.backend.vendor}")
    print(f"  Device             : {info.name}")
    print(f"  Torch device       : {info.device}")
    print(f"  Autocast type      : {info.autocast_device_type}")
    if info.compute_capability:
        print(f"  Architecture       : {info.compute_capability}")
    if info.total_memory:
        print(f"  Memory             : {info.total_memory / 1024**3:.2f} GB")
    print(f"  supports_amp       : {supports_amp(device)}")
    print(f"  supports_scaler    : {supports_grad_scaler(device)}")

    _, warning = resolve_amp_settings(device, use_amp=True)
    if warning:
        print(f"  AMP note           : {warning.strip()}")

    for note in info.notes:
        print(f"  Note               : {note}")

    if torch.cuda.is_available():
        print()
        print("  torch.cuda report")
        try:
            print(f"    device_count     : {torch.cuda.device_count()}")
            print(f"    current_device   : {torch.cuda.current_device()}")
            print(f"    get_device_name  : {torch.cuda.get_device_name(info.device.index or 0)}")
            print(f"    capability       : {torch.cuda.get_device_capability(info.device.index or 0)}")
            properties = torch.cuda.get_device_properties(info.device.index or 0)
            for attribute in ("major", "minor", "gcnArchName", "multi_processor_count"):
                if hasattr(properties, attribute):
                    print(f"    {attribute:<17}: {getattr(properties, attribute)}")
            print(f"    cudnn available  : {torch.backends.cudnn.is_available()}")
        except Exception as error:  # pragma: no cover - driver dependent
            print(f"    (failed to query: {error!r})")


def run_functional_checks(device: torch.device) -> list[tuple[str, bool, str]]:
    """
    Exercise the ops HanziGen actually uses: conv2d, conv_transpose2d, group
    norm, SiLU/Tanh, matmul, embedding+argmin, and one autograd step.
    """
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            detail = fn()
            results.append((name, True, detail or "ok"))
        except Exception as error:
            results.append((name, False, f"{type(error).__name__}: {error}"))

    def _conv2d() -> str:
        layer = torch.nn.Conv2d(1, 8, 4, stride=2, padding=1).to(device)
        out = layer(torch.randn(2, 1, 32, 32, device=device))
        return f"out={tuple(out.shape)} dtype={out.dtype}"

    def _conv_transpose2d() -> str:
        layer = torch.nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1).to(device)
        out = layer(torch.randn(2, 8, 8, 8, device=device))
        return f"out={tuple(out.shape)}"

    def _group_norm_silu() -> str:
        layer = torch.nn.Sequential(
            torch.nn.GroupNorm(4, 8),
            torch.nn.SiLU(),
            torch.nn.Tanh(),
        ).to(device)
        out = layer(torch.randn(2, 8, 16, 16, device=device))
        return f"out={tuple(out.shape)}"

    def _matmul_distances() -> str:
        a = torch.randn(256, 2, device=device)
        b = torch.randn(64, 2, device=device)
        distances = torch.sum(a**2, dim=1, keepdim=True) + torch.sum(b**2, dim=1) - 2 * (a @ b.t())
        indices = torch.argmin(distances, dim=1)
        return f"indices={tuple(indices.shape)} max={int(indices.max().item())}"

    def _embedding() -> str:
        embedding = torch.nn.Embedding(64, 2).to(device)
        out = embedding(torch.randint(0, 64, (16,), device=device))
        return f"out={tuple(out.shape)}"

    def _backward_step() -> str:
        layer = torch.nn.Conv2d(1, 8, 3, padding=1).to(device)
        optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
        loss = layer(torch.randn(2, 1, 16, 16, device=device)).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return f"loss={loss.item():.4f}"

    def _amp_float16() -> str:
        if not supports_amp(device):
            return "skipped (backend has no autocast)"
        from utils.hardware.hardware_utils import autocast_context

        layer = torch.nn.Conv2d(1, 8, 3, padding=1).to(device)
        with autocast_context(device, enabled=True):
            out = layer(torch.randn(2, 1, 16, 16, device=device))
        return f"out dtype={out.dtype}"

    def _grad_scaler() -> str:
        from utils.hardware.hardware_utils import make_grad_scaler

        scaler = make_grad_scaler(device=device, enabled=True)
        layer = torch.nn.Conv2d(1, 8, 3, padding=1).to(device)
        optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
        loss = layer(torch.randn(2, 1, 16, 16, device=device)).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        return f"enabled={getattr(scaler, 'enabled', False)}"

    def _host_transfer() -> str:
        tensor = torch.randn(2, 1, 16, 16, device=device)
        back = tensor.detach().cpu()
        return f"roundtrip dtype={back.dtype}"

    print()
    print("=" * 72)
    print("Functional checks")
    print("=" * 72)

    check("conv2d (stride 2 down)", _conv2d)
    check("conv_transpose2d (stride 2 up)", _conv_transpose2d)
    check("group_norm + silu + tanh", _group_norm_silu)
    check("codebook distance + argmin", _matmul_distances)
    check("embedding lookup", _embedding)
    check("autograd step (backward + Adam)", _backward_step)
    check("float16 autocast", _amp_float16)
    check("grad scaler step", _grad_scaler)
    check("device -> host transfer", _host_transfer)

    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name:<32} {detail}")

    return results


def run_benchmark(device: torch.device, iterations: int = 10) -> None:
    """
    Measure a conv + groupnorm forward/backward loop that matches the VQ-VAE
    training step in this project (512 x 512 glyph images, batch 8).

    A smaller workload would be dominated by kernel launch overhead and would
    give misleading numbers on a GPU.
    """
    print()
    print("=" * 72)
    print("Benchmark (VQ-VAE-sized conv + groupnorm, forward + backward)")
    print("=" * 72)
    print(f"  Device: {describe_device(device)}")

    batch, width = 8, 512
    print(f"  Shape : batch={batch}, 1x{width}x{width}")
    print(f"  Model : Conv2d(1->64,s2) -> Conv2d(64->128,s2) -> Conv2d(128->256,s2)")

    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, 4, stride=2, padding=1),
        torch.nn.GroupNorm(8, 64),
        torch.nn.SiLU(),
        torch.nn.Conv2d(64, 128, 4, stride=2, padding=1),
        torch.nn.GroupNorm(16, 128),
        torch.nn.SiLU(),
        torch.nn.Conv2d(128, 256, 4, stride=2, padding=1),
        torch.nn.GroupNorm(32, 256),
        torch.nn.SiLU(),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    def step() -> None:
        images = torch.randn(batch, 1, width, width, device=device)
        loss = model(images).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    try:
        # Warm-up, excluded from the measurement.
        for _ in range(3):
            step()
        synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            step()
        synchronize()
        elapsed = time.perf_counter() - start
    except RuntimeError as error:
        print(f"  [FAIL] benchmark could not run: {type(error).__name__}: {error}")
        return

    per_step = elapsed / iterations
    print()
    print(f"  Iterations         : {iterations}")
    print(f"  Total              : {elapsed:.3f} s")
    print(f"  Per step           : {per_step * 1000:.2f} ms")
    print(f"  Throughput         : {batch / per_step:.1f} images/s")
    print()
    print("  Measured reference points for this exact workload:")
    print("    AMD Ryzen 7000/9000, CPU only  : ~699 ms per step ( 11 img/s)")
    print("    AMD RX 9070 XT, DirectML       : ~171 ms per step ( 47 img/s)")
    print("    AMD RX 9070 XT, ROCm / HIP     :  ~42 ms per step (191 img/s)")
    print("    NVIDIA RTX 30/40 series (CUDA) : tens of ms per step")
    print()
    print("  ROCm is the only AMD path with torch.amp support, and it is ~4x")
    print("  faster than DirectML here. DirectML falls back to the CPU for")
    print("  unimplemented operators (torch warns when it does) and cannot use")
    print("  mixed precision, so treat it as a compatibility fallback.")
    print("  If you are slower than the CPU figure above, the run is probably on")
    print("  the CPU: check the 'Backend' line.")


def print_installed_ops(device: torch.device) -> None:
    info = get_device_info(device)
    key = f"{info.autocast_device_type}." if info.device.type == "cuda" else f"{info.device.type}."

    try:
        registry = torch._C._jit_get_all_schemas()
    except Exception as error:  # pragma: no cover
        print(f"  (could not enumerate ops: {error!r})")
        return

    names = sorted({str(schema.name) for schema in registry})
    conv_ops = [name for name in names if "conv" in name.lower()]
    norm_ops = [name for name in names if "norm" in name.lower()]

    print()
    print("=" * 72)
    print(f"Registered convolution ops ({len(conv_ops)} of {len(names)} total)")
    print("=" * 72)
    for name in conv_ops:
        print(f"  {name}")

    print()
    print(f"Registered normalization ops ({len(norm_ops)})")
    for name in norm_ops:
        print(f"  {name}")


def main() -> None:
    args = parse_args()

    print_environment()

    try:
        device = select_device(args.device)
    except RuntimeError as error:
        print(f"\n❌ {error}")
        sys.exit(1)

    print_selected_device(device)
    results = run_functional_checks(device)

    if args.benchmark:
        run_benchmark(device)
    if args.installed_ops:
        print_installed_ops(device)

    failures = [name for name, ok, _ in results if not ok]

    print()
    print("=" * 72)
    if failures:
        print(f"Result: {len(failures)} check(s) FAILED -> {', '.join(failures)}")
        print("See docs/AMD_SETUP.md for the recommended backend for your GPU.")
    else:
        print("Result: all checks passed. HanziGen can run on this backend.")
    print("=" * 72)

    if failures and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
