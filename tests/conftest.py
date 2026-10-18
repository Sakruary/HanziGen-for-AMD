"""
Shared pytest configuration.

Two pieces of process-wide state leak between tests in this project, so both
are handled here rather than in each test module.
"""

from __future__ import annotations

import pytest
import torch

from utils.hardware import backends


@pytest.fixture(autouse=True)
def _isolate_device_cache():
    """
    Guarantee that each test starts from a clean detection cache.

    ``utils.hardware.backends`` memoises auto-detection, which is desirable in
    production but leaks between tests: a test that simulates a ROCm machine
    would otherwise be observed by the next test. Resetting around every test
    makes the suite order-independent.
    """
    backends.reset_device_cache()
    yield
    backends.reset_device_cache()


@pytest.fixture(scope="session", autouse=True)
def _initialise_accelerator_autograd():
    """
    Touch the accelerator with one real forward/backward before any test runs.

    PyTorch's autograd engine sizes its per-device "ready queues" when it is
    first used. If the very first backward pass in the process is a CPU one,
    the queue for the accelerator's device type is created with zero entries,
    and the first accelerator backward then dies with::

        RuntimeError: 0 <= device.index() && device.index() <
        static_cast<c10::DeviceIndex>(device_ready_queues_.size())
        INTERNAL ASSERT FAILED at ".../autograd/engine.cpp"

    That is reproducible with torch-directml on Windows and has nothing to do
    with the test that happens to run first: any CPU autograd work (a metrics
    pass, a small utility model) before the first GPU step would trigger it.
    Running a trivial accelerator backward here makes the ordering explicit.

    Every failure is swallowed: on a CPU-only or unsupported backend this is a
    no-op, and it must never be the reason a test session fails.
    """
    try:
        from utils.hardware.hardware_utils import select_device

        device = select_device()
        if device.type == "cpu":
            return

        parameter = torch.nn.Parameter(torch.ones(2, device=device))
        (parameter * 2).sum().backward()
    except Exception:  # pragma: no cover - backend dependent, intentionally ignored
        return
