"""
Backend selection tests.

These run on any machine: they simulate the presence or absence of CUDA, ROCm,
ZLUDA, DirectML and MPS by patching the probes that
``utils.hardware.backends`` uses, then assert that the right backend and torch
device are selected.

Run with:
    uv run pytest tests/test_backend_selection.py -v
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from utils.hardware import backends
from utils.hardware.backends import HardwareBackend, normalize_device


@contextlib.contextmanager
def simulated(
    cuda: bool = False,
    hip: str | None = None,
    mps: bool = False,
    directml: bool = False,
    zluda: bool = False,
    rocm_masquerade: bool = False,
):
    """
    Simulate a machine that has (only) the requested accelerators.

    Every probe inside ``utils.hardware.backends`` is patched, so this works
    identically on a CUDA box, an AMD box and a CPU-only CI runner.
    """
    device_name = "AMD Radeon RX 9070 XT" if (zluda or rocm_masquerade) else "NVIDIA GeForce RTX 4090"
    device_properties = SimpleNamespace(
        name=device_name,
        major=0 if (zluda or rocm_masquerade) else 8,
        minor=9 if not (zluda or rocm_masquerade) else 0,
        total_memory=16 * 1024**3,
        gcnArchName="gfx1201" if rocm_masquerade else "",
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda, raising=False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps, raising=False)
        monkeypatch.setattr(torch.backends.mps, "is_built", lambda: mps, raising=False)
        monkeypatch.setattr(
            backends, "_import_directml", lambda *a, **k: object() if directml else None
        )
        monkeypatch.setattr(backends, "_detect_zluda", lambda index=0: zluda)
        monkeypatch.setattr(
            backends, "_detect_rocm_masquerade", lambda index=0: rocm_masquerade
        )
        monkeypatch.setattr(backends, "_cuda_device_name", lambda index=0: device_name)
        monkeypatch.setattr(
            backends, "_cuda_device_properties", lambda index=0: device_properties
        )
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda index=0: (0, 0)
        )

        original_hip = torch.version.hip
        monkeypatch.setattr(torch.version, "hip", hip, raising=False)
        monkeypatch.delenv(backends.BACKEND_ENV_VAR, raising=False)
        backends.reset_device_cache()

        try:
            yield monkeypatch
        finally:
            monkeypatch.setattr(torch.version, "hip", original_hip, raising=False)
            backends.reset_device_cache()


def test_auto_detects_nvidia_cuda():
    with simulated(cuda=True):
        info = backends.detect_backend()
        assert info.backend is HardwareBackend.CUDA
        assert info.backend.vendor == "NVIDIA"
        assert info.device.type == "cuda"
        assert info.autocast_device_type == "cuda"


def test_auto_detects_amd_rocm():
    with simulated(cuda=True, hip="6.3.42131"):
        info = backends.detect_backend()
        assert info.backend is HardwareBackend.ROCM
        assert info.backend.vendor == "AMD"
        # ROCm devices are addressed through the cuda namespace ...
        assert info.device.type == "cuda"
        # ... and autocast must be told "cuda", not "rocm".
        assert info.autocast_device_type == "cuda"


def test_auto_detects_amd_zluda():
    with simulated(cuda=True, zluda=True):
        info = backends.detect_backend()
        assert info.backend is HardwareBackend.ZLUDA
        assert info.backend.vendor == "AMD"
        assert info.device.type == "cuda"


def test_auto_detects_rocm_masquerade():
    """
    A ROCm-on-Windows build that patches the CUDA namespace without setting
    ``torch.version.hip`` must still be recognised as AMD, not ZLUDA.
    """
    with simulated(cuda=True, rocm_masquerade=True):
        info = backends.detect_backend()
        assert info.backend is HardwareBackend.ROCM
        assert info.backend.vendor == "AMD"


def test_zluda_probe_uses_compute_capability_not_only_the_name():
    """
    Exercise the real probe (``_detect_zluda`` is not stubbed here) to confirm
    it keys off an unusable compute capability rather than the device name.
    """
    without_sm = SimpleNamespace(major=0, minor=0, total_memory=1, gcnArchName="")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(backends, "_cuda_device_name", lambda index=0: "Generic GPU")
        monkeypatch.setattr(backends, "_cuda_device_properties", lambda index=0: without_sm)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (0, 0))
        assert backends._detect_zluda() is True

    with_nvidia_sm = SimpleNamespace(major=8, minor=9, total_memory=1, gcnArchName="")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(backends, "_cuda_device_name", lambda index=0: "NVIDIA GeForce RTX 4090")
        monkeypatch.setattr(backends, "_cuda_device_properties", lambda index=0: with_nvidia_sm)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (8, 9))
        assert backends._detect_zluda() is False


def test_rocm_masquerade_probe_uses_gcn_arch_name():
    """
    ``gcnArchName`` is a HIP-only property, so its presence identifies a
    ROCm-based CUDA namespace rather than ZLUDA.
    """
    hip_like = SimpleNamespace(major=0, minor=0, total_memory=1, gcnArchName="gfx1201")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(backends, "_cuda_device_name", lambda index=0: "AMD Radeon RX 9070 XT")
        monkeypatch.setattr(backends, "_cuda_device_properties", lambda index=0: hip_like)
        assert backends._detect_rocm_masquerade() is True

    zluda_like = SimpleNamespace(major=0, minor=0, total_memory=1, gcnArchName="")
    with pytest.MonkeyPatch.context() as monkeypatch:
        # Without gcnArchName this case is genuinely ambiguous; the AMD device
        # name plus the absent architecture is what tips it to ROCm here.
        monkeypatch.setattr(backends, "_cuda_device_name", lambda index=0: "AMD Radeon RX 9070 XT")
        monkeypatch.setattr(backends, "_cuda_device_properties", lambda index=0: zluda_like)
        monkeypatch.setattr(backends, "_cuda_compute_capability", lambda index=0: None)
        assert backends._detect_rocm_masquerade() is True

    nvidia_like = SimpleNamespace(major=8, minor=9, total_memory=1, gcnArchName="")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(backends, "_cuda_device_name", lambda index=0: "NVIDIA GeForce RTX 4090")
        monkeypatch.setattr(backends, "_cuda_device_properties", lambda index=0: nvidia_like)
        monkeypatch.setattr(backends, "_cuda_compute_capability", lambda index=0: "8.9")
        assert backends._detect_rocm_masquerade() is False


def test_auto_detects_directml():
    with simulated(directml=True):
        info = backends.detect_backend()
        assert info.backend is HardwareBackend.DIRECTML
        assert info.device.type == "privateuseone"
        assert info.autocast_device_type == "cpu"


def test_auto_detects_mps():
    with simulated(mps=True):
        info = backends.detect_backend()
        assert info.backend is HardwareBackend.MPS
        assert info.device.type == "mps"


def test_auto_falls_back_to_cpu():
    with simulated():
        info = backends.detect_backend()
        assert info.backend is HardwareBackend.CPU
        assert info.device.type == "cpu"


def test_cpu_wins_over_no_accelerator_and_never_claims_amp():
    with simulated():
        assert not backends.supports_amp(None)
        assert not backends.supports_grad_scaler(None)


def test_directml_disables_amp_and_scaler():
    with simulated(directml=True):
        assert not backends.supports_amp(None)
        assert not backends.supports_grad_scaler(None)

        scaler = backends.make_grad_scaler(None, enabled=True)
        assert scaler.enabled is False

        with backends.autocast_context(None, enabled=True) as unused:
            assert unused is None


def test_rocm_supports_amp_and_scaler():
    with simulated(cuda=True, hip="6.3.42131"):
        assert backends.supports_amp(None)
        assert backends.supports_grad_scaler(None)


class TestNormalizeDevice:
    def test_none_auto_selects(self):
        with simulated(cuda=True):
            device, backend = normalize_device(None)
            assert device.type == "cuda"
            assert backend is None

    @pytest.mark.parametrize("alias", ["rocm", "hip", "amd"])
    def test_rocm_aliases_map_to_cuda_device(self, alias):
        with simulated(cuda=True, hip="6.3.42131"):
            device, backend = normalize_device(alias)
            assert backend is HardwareBackend.ROCM
            assert device.type == "cuda"

    @pytest.mark.parametrize("alias", ["directml", "dml"])
    def test_directml_aliases(self, alias):
        with simulated(directml=True):
            device, backend = normalize_device(alias)
            assert backend is HardwareBackend.DIRECTML
            assert device.type == "privateuseone"

    def test_index_is_preserved(self):
        with simulated(cuda=True, hip="6.3.42131"):
            device, backend = normalize_device("rocm:1")
            assert device.index == 1
            assert backend is HardwareBackend.ROCM

    def test_cpu_alias(self):
        device, backend = normalize_device("cpu")
        assert device.type == "cpu"
        assert backend is HardwareBackend.CPU

    def test_unknown_string_is_forwarded_to_torch(self):
        with simulated():
            with pytest.raises(RuntimeError):
                # torch.device rejects unknown types.
                normalize_device("not-a-device")

    def test_requesting_rocm_without_cuda_raises(self):
        with simulated():
            with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
                normalize_device("rocm")

    def test_requesting_directml_without_module_raises(self):
        with simulated():
            with pytest.raises(RuntimeError, match="torch_directml"):
                normalize_device("directml")


def test_env_override_selects_backend():
    with simulated(cuda=True, hip="6.3.42131") as monkeypatch:
        monkeypatch.setenv(backends.BACKEND_ENV_VAR, "cpu")
        info = backends.get_device_info(None)
        assert info.backend is HardwareBackend.CPU


def test_env_override_auto_is_ignored():
    with simulated(cuda=True):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv(backends.BACKEND_ENV_VAR, "auto")
            backends.reset_device_cache()
            info = backends.get_device_info(None)
            assert info.backend is HardwareBackend.CUDA


def test_describe_device_is_human_readable():
    with simulated(cuda=True, hip="6.3.42131", rocm_masquerade=True):
        description = backends.describe_device(None)
        assert "ROCm" in description
        assert "RX 9070 XT" in description
        # VRAM is reported in GB and the architecture is included.
        assert "GB" in description
        assert "gfx1201" in description


def test_describe_device_for_cpu_is_short():
    with simulated():
        assert backends.describe_device(None) == "CPU"


def test_grad_scaler_fallback_is_drop_in_compatible():
    with simulated():
        scaler = backends.make_grad_scaler(None, enabled=False)
        assert isinstance(scaler, backends._GradScalerFallback)
        assert scaler.enabled is False
        assert scaler.is_enabled() is False

        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        loss = parameter.pow(2).sum()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        assert parameter.grad is not None
        assert scaler.get_scale() == 1.0
        assert scaler.state_dict()["enabled"] is False


class TestScalerIsEnabled:
    """
    ``torch.amp.GradScaler`` exposes ``is_enabled()`` and has **no** ``enabled``
    attribute. Reading ``getattr(scaler, "enabled", False)`` therefore always
    reports False for a real scaler, which silently turned autocast off even
    when the user asked for AMP. These tests pin the correct behaviour.
    """

    def test_none_scaler_is_disabled(self):
        assert backends.scaler_is_enabled(None) is False

    def test_real_grad_scaler_is_recognised(self):
        with simulated():
            fallback = backends.make_grad_scaler(None, enabled=False)
        assert backends.scaler_is_enabled(fallback) is False

        real = torch.amp.GradScaler("cpu", enabled=True)
        # The whole point: the attribute the old code read does not exist ...
        assert not hasattr(real, "enabled")
        # ... while is_enabled() reports the truth.
        assert real.is_enabled() is True
        assert backends.scaler_is_enabled(real) is True

    def test_integer_and_truthy_values_are_coerced(self):
        class Fake:
            enabled = 1

        assert backends.scaler_is_enabled(Fake()) is True

    def test_is_enabled_taking_an_unexpected_argument_is_survivable(self):
        class Broken:
            def is_enabled(self, required_argument):  # pragma: no cover
                return True

        assert backends.scaler_is_enabled(Broken()) is False
