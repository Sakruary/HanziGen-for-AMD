"""
Tests for the backend-aware launcher (``run.py``).

``run.py`` decides which backend to use before any project module is imported,
so it is worth pinning down separately from the detection tests.

Run with:
    uv run pytest tests/test_run_launcher.py -v
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from utils.hardware.backends import BACKEND_ENV_VAR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_PATH = PROJECT_ROOT / "run.py"


@pytest.fixture
def marker_dir():
    """
    A throwaway directory inside the repository.

    ``tmp_path`` is avoided deliberately: the system temp directory is not
    always writable (restricted or sandboxed environments), and the constraints
    lookup only cares about file existence.
    """
    import shutil

    directory = PROJECT_ROOT / ".scratch" / "pytest-marker-dir"
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def load_launcher():
    """
    Import ``run.py`` as a module without executing ``main()``.

    The module is registered in ``sys.modules`` before execution because
    ``run.py`` defines a dataclass, and ``dataclasses`` resolves the defining
    module through ``sys.modules`` while decorating it.
    """
    import sys

    name = "hanzigen_run_launcher"
    spec = importlib.util.spec_from_file_location(name, LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@pytest.fixture(scope="module")
def launcher():
    return load_launcher()


class TestBackendResolution:
    def test_explicit_backend_wins(self, launcher):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv(BACKEND_ENV_VAR, "cpu")
            resolution = launcher.resolve_backend("rocm")
        assert resolution.backend == "rocm"
        assert "explicit" in resolution.reason

    def test_environment_variable_is_honoured(self, launcher):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv(BACKEND_ENV_VAR, "directml")
            resolution = launcher.resolve_backend(None)
        assert resolution.backend == "directml"
        assert BACKEND_ENV_VAR in resolution.reason

    def test_auto_ignores_auto_environment_value(self, launcher):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv(BACKEND_ENV_VAR, "auto")
            resolution = launcher.resolve_backend(None)
        assert resolution.backend != "auto"
        assert resolution.backend in {"cuda", "rocm", "mps", "directml", "cpu"}

    def test_unspecified_backend_never_returns_auto(self, launcher):
        """
        The launcher must always produce a concrete device for ``--device`` so
        that the target script does not have to re-detect anything.
        """
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.delenv(BACKEND_ENV_VAR, raising=False)
            resolution = launcher.resolve_backend(None)
        assert resolution.backend in {"cuda", "rocm", "mps", "directml", "cpu"}

    def test_constraints_marker_is_used_only_when_unambiguous(self, launcher, marker_dir, monkeypatch):
        fakes = {
            "amd-rocm.txt": "rocm",
            "amd-zluda.txt": "zluda",
            "amd-directml.txt": "directml",
            "cpu.txt": "cpu",
            "nvidia-cuda.txt": "cuda",
        }
        for name in fakes:
            (marker_dir / name).write_text("", encoding="utf-8")

        monkeypatch.setattr(launcher, "CONSTRAINTS_DIR", marker_dir)
        monkeypatch.delenv(BACKEND_ENV_VAR, raising=False)

        # All five markers exist, so the choice is ambiguous and detection wins.
        resolution = launcher.resolve_backend(None)
        assert "only backend marker" not in resolution.reason

        # Leaving a single marker makes it authoritative.
        for name in list(fakes)[1:]:
            (marker_dir / name).unlink()
        resolution = launcher.resolve_backend(None)
        assert resolution.backend == "rocm"
        assert "amd-rocm.txt" in resolution.reason

    def test_missing_markers_do_not_crash(self, launcher, marker_dir, monkeypatch):
        monkeypatch.setattr(launcher, "CONSTRAINTS_DIR", marker_dir / "does-not-exist")
        with pytest.MonkeyPatch.context() as inner:
            inner.delenv(BACKEND_ENV_VAR, raising=False)
            resolution = launcher.resolve_backend(None)
        assert resolution.backend in {"cuda", "rocm", "mps", "directml", "cpu"}


class TestDeviceArgument:
    def test_cpu_maps_to_cpu(self, launcher):
        device, detail = launcher._resolve_device_argument("cpu")
        assert device == "cpu"
        assert "CPU" in detail

    def test_mps_maps_to_mps(self, launcher):
        device, _ = launcher._resolve_device_argument("mps")
        assert device == "mps"

    def test_auto_defers_to_pytorch(self, launcher):
        device, detail = launcher._resolve_device_argument("auto")
        assert device == "auto"
        assert "deferred" in detail

    def test_requesting_cuda_on_a_cpu_build_raises(self, launcher):
        import torch

        if torch.cuda.is_available():
            pytest.skip("this machine really has CUDA")
        with pytest.raises(RuntimeError, match="CUDA"):
            launcher._resolve_device_argument("cuda")

    def test_requesting_rocm_without_a_hip_build_raises(self, launcher):
        import torch

        if torch.version.hip is not None and torch.cuda.is_available():
            pytest.skip("this machine really has ROCm")

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv(launcher.NO_INSTALL_ENV_VAR, "1")
            with pytest.raises(RuntimeError) as error:
                launcher._resolve_device_argument("rocm")

        assert "ROCm" in str(error.value)

    def test_requesting_directml_without_the_package_raises(self, launcher):
        if importlib.util.find_spec("torch_directml") is not None:
            pytest.skip("torch_directml is installed here")
        with pytest.raises(RuntimeError, match="torch_directml"):
            launcher._resolve_device_argument("directml")


class TestArguments:
    def test_parser_accepts_every_documented_backend(self, launcher):
        for backend in launcher.KNOWN_BACKENDS:
            namespace = launcher.build_parser().parse_args(["--backend", backend, "train_vqvae.py"])
            assert namespace.backend == backend

    def test_parser_accepts_an_empty_backend(self, launcher):
        """
        The wrappers pass ``--backend ""`` when DEVICE is left empty; argparse
        must not reject it (which is why `choices` is not used).
        """
        namespace = launcher.build_parser().parse_args(["--backend", "", "train_vqvae.py"])
        assert namespace.backend == ""
        assert launcher._normalize_backend_value(namespace.backend) is None

    def test_list_and_script_can_be_combined(self, launcher):
        namespace = launcher.build_parser().parse_args(["--list", "train_vqvae.py"])
        assert namespace.list is True
        assert namespace.script == "train_vqvae.py"

    def test_unknown_backends_are_rejected_by_validation(self, launcher):
        """
        ``main`` compares the normalised value against KNOWN_BACKENDS, so an
        unknown name must fall outside that set.
        """
        for bad in ("banana", "cudaa", "gpu"):
            value = launcher._normalize_backend_value(bad)
            assert value not in launcher.KNOWN_BACKENDS

    def test_empty_backend_is_within_known_values(self, launcher):
        assert "auto" in launcher.KNOWN_BACKENDS
        assert launcher._normalize_backend_value("") is None


class TestEmptyBackendValue:
    """
    The wrapper scripts leave DEVICE empty by default, which turns into
    ``--backend ""``. That has to mean "auto-detect", not "invalid".
    """

    def test_empty_string_is_normalised_to_none(self, launcher):
        assert launcher._normalize_backend_value("") is None
        assert launcher._normalize_backend_value("   ") is None
        assert launcher._normalize_backend_value(None) is None

    def test_real_values_survive_normalisation(self, launcher):
        assert launcher._normalize_backend_value(" rocm ") == "rocm"
        assert launcher._normalize_backend_value("directml") == "directml"

    def test_empty_backend_falls_back_to_detection(self, launcher):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.delenv(BACKEND_ENV_VAR, raising=False)
            resolution = launcher.resolve_backend(launcher._normalize_backend_value(""))
        # On a GPU box this detects the GPU; on a CPU box it detects CPU. Either
        # way it must not raise and must not stay "auto".
        assert resolution.backend in {"cuda", "rocm", "mps", "directml", "cpu"}

    def test_auto_backend_never_constructs_a_device(self, launcher):
        """
        ``auto`` must not probe anything: that is what makes it safe for the
        CPU-only data-preparation scripts.
        """
        device, detail = launcher._resolve_device_argument("auto")
        assert device == "auto"
        assert "deferred" in detail


class TestWrapperScripts:
    """
    The .bat/.sh wrappers must go through run.py, otherwise the backend
    resolution they advertise would not happen.
    """

    #: Wrappers for scripts that accept --device, so they need the launcher.
    DEVICE_AWARE = {
        "train_vqvae",
        "train_ldm",
        "split_dataset",
        "compute_metrics",
        "inference",
    }

    #: Wrappers for pure data-preparation scripts. These never touch a GPU, so
    #: running them directly (under any backend) is intentional.
    DEVICE_FREE = {"analyze_font", "prepare_dataset", "convert_to_svg"}

    def _wrappers(self) -> list[Path]:
        paths = sorted((PROJECT_ROOT / "scripts" / "bat").glob("*.bat"))
        paths += sorted((PROJECT_ROOT / "scripts" / "sh").glob("*.sh"))
        # pipeline.* only calls the other wrappers.
        return [path for path in paths if path.stem != "pipeline"]

    def test_wrappers_are_fully_classified(self):
        """
        Guard against a new wrapper being added without a decision about
        whether it needs backend resolution.
        """
        found = {path.stem for path in self._wrappers()}
        classified = self.DEVICE_AWARE | self.DEVICE_FREE
        assert found == classified, (
            f"unclassified wrappers: {sorted(found - classified)}; "
            f"missing wrappers: {sorted(classified - found)}"
        )

    def test_device_aware_wrappers_call_the_launcher(self):
        offenders = [
            path.name
            for path in self._wrappers()
            if path.stem in self.DEVICE_AWARE
            and "run.py" not in path.read_text(encoding="utf-8", errors="replace")
        ]
        assert not offenders, f"wrappers do not use run.py: {offenders}"

    def test_device_aware_wrappers_do_not_inject_device_themselves(self):
        """
        run.py owns --device; a wrapper passing its own would win and bypass
        resolution. Comment lines are ignored.
        """
        pattern = re.compile(r"^\s*(?!REM\b|#).*--device", re.IGNORECASE)
        for path in self._wrappers():
            if path.stem not in self.DEVICE_AWARE:
                continue
            offenders = [
                line
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if pattern.search(line)
            ]
            assert not offenders, f"{path.name} passes --device itself: {offenders}"

    def test_device_aware_wrappers_do_not_pass_an_empty_backend(self):
        """
        Passing ``--backend ""`` relies on the shell and the batch interpreter
        both preserving an empty argument, which they do not agree on. The
        wrappers therefore set HANZIGEN_BACKEND instead.
        """
        for path in self._wrappers():
            text = path.read_text(encoding="utf-8", errors="replace")
            assert "--backend" not in text, f"{path.name} still passes --backend"

    def test_wrappers_document_the_environment_variable(self):
        offenders = [
            path.name
            for path in self._wrappers()
            if path.stem in self.DEVICE_AWARE
            and BACKEND_ENV_VAR not in path.read_text(encoding="utf-8", errors="replace")
        ]
        assert not offenders, f"wrappers do not document {BACKEND_ENV_VAR}: {offenders}"

    def test_wrappers_do_not_hardcode_a_cuda_device(self):
        offenders = []
        for path in self._wrappers():
            text = path.read_text(encoding="utf-8", errors="replace")
            if re.search(r'DEVICE\s*=\s*"?cuda"?', text):
                offenders.append(path.name)
        assert not offenders, f"wrappers still hardcode cuda: {offenders}"
