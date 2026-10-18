"""
End-to-end training-step tests.

These build the real VQ-VAE and LDM models and run one training batch plus one
validation batch through ``_process_batch``, on whatever backend the machine
provides. They cover the exact code paths that were changed for AMD support:

* the merged ``scaler is not None`` branch,
* ``autocast_context`` on a backend that may or may not support autocast,
* ``make_grad_scaler`` returning either a real scaler or the no-op fallback,
* the float32 codebook lookup inside ``autocast``.

The images are deliberately tiny (64x64, batch 2) so the suite stays fast even
on CPU, while still respecting the architecture: the VQ-VAE halves the
resolution three times, and the UNet halves it three more times, so anything
smaller than 64x64 cannot be reconstructed.

Run with:

    uv run pytest tests/test_training_step.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from configs import LDMModelConfig, VQVAEModelConfig
from models import LDM, VQVAE
from utils.hardware import backends
from utils.hardware.hardware_utils import (
    load_state_dict,
    make_grad_scaler,
    save_state_dict,
    select_device,
    supports_grad_scaler,
)

#: 64 = 2**6, which is the smallest size that survives the VQ-VAE's three
#: stride-2 downsamples (64 -> 32 -> 16 -> 8) and the UNet's three more.
IMAGE_SIZE = 64
BATCH_SIZE = 2

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def device() -> torch.device:
    return select_device()


@pytest.fixture
def batch(device: torch.device) -> dict[str, torch.Tensor | list[str]]:
    """
    A batch shaped exactly like ``PairedGlyphImageDataset`` produces, already
    normalised to [-1, 1] and placed on the selected device.
    """
    return {
        "tgt_img": (torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE) * 2 - 1).to(device),
        "ref_img": (torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE) * 2 - 1).to(device),
        "img_name": [f"glyph_{index}" for index in range(BATCH_SIZE)],
    }


@pytest.fixture(scope="module")
def vqvae_config() -> VQVAEModelConfig:
    # Tiny channels keep the test fast on CPU.
    return VQVAEModelConfig(
        input_img_channels=1,
        encoder_base_channels=8,
        latent_dim=2,
        codebook_size=16,
        commitment_cost=0.25,
    )


@pytest.fixture(scope="module")
def ldm_config() -> LDMModelConfig:
    return LDMModelConfig(
        unet_base_channels=8,
        time_pos_dim=32,
        time_emb_dim=64,
        time_steps=100,
    )


def test_vqvae_training_step_without_amp(device, vqvae_config, batch):
    model = VQVAE(model_config=vqvae_config, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = model._process_batch(
        batch=batch,
        is_training=True,
        optimizer=optimizer,
        scaler=None,
    )

    assert set(losses) == {"total", "recon", "vq"}
    for name, value in losses.items():
        assert torch.isfinite(torch.tensor(value)), f"non-finite {name} loss: {losses}"

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients were produced"
    assert all(torch.isfinite(g).all() for g in grads)


def test_vqvae_training_step_with_grad_scaler(device, vqvae_config, batch):
    """
    ``make_grad_scaler`` must yield something usable on every backend: a real
    scaler where supported, the no-op fallback otherwise.
    """
    model = VQVAE(model_config=vqvae_config, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scaler = make_grad_scaler(device=device, enabled=True)
    assert hasattr(scaler, "scale")

    losses = model._process_batch(
        batch=batch,
        is_training=True,
        optimizer=optimizer,
        scaler=scaler,
    )

    assert torch.isfinite(torch.tensor(losses["total"]))
    assert any(p.grad is not None for p in model.parameters())


def test_vqvae_validation_step_does_not_update_weights(device, vqvae_config, batch):
    model = VQVAE(model_config=vqvae_config, device=device)
    before = [p.detach().clone() for p in model.parameters()]

    losses = model._process_batch(
        batch=batch,
        is_training=False,
        optimizer=None,
        scaler=None,
    )

    assert torch.isfinite(torch.tensor(losses["total"]))
    after = [p.detach() for p in model.parameters()]
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_quantizer_distances_are_float32_under_autocast(device):
    """
    The codebook lookup runs inside ``no_autocast``, so it must not raise the
    "expected scalar type Half but found Float" error that a naive autocast
    implementation produces, and it must stay numerically correct.
    """
    from models.vqvae.vqvae_encoder_decoder import VQVAEQuantizer
    from utils.hardware.hardware_utils import autocast_context

    quantizer = VQVAEQuantizer(codebook_size=16, latent_dim=2, commitment_cost=0.25).to(device)
    latents = torch.randn(2, 2, 4, 4, device=device)

    with autocast_context(device, enabled=True):
        quantized, loss = quantizer(latents)

    assert quantized.shape == latents.shape
    assert torch.isfinite(loss)
    # Under autocast the quantizer output matches the (possibly half) input dtype.
    assert quantized.dtype == latents.dtype

    # The lookup itself must be identical to a pure float32 run.
    reference_quantizer = VQVAEQuantizer(
        codebook_size=16, latent_dim=2, commitment_cost=0.25
    ).to(device)
    with torch.no_grad():
        reference_quantizer.emb.weight.copy_(quantizer.emb.weight)
    reference, _ = reference_quantizer(latents.float())
    assert torch.allclose(quantized.float(), reference, atol=1e-4)


def test_ldm_training_step(device, vqvae_config, ldm_config, batch):
    model = LDM(
        vqvae_model_config=vqvae_config,
        ldm_model_config=ldm_config,
        device=device,
    )
    model._freeze_vqvae()

    # Only the UNet should be optimised, mirroring train_ldm.py.
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "the UNet has no trainable parameters"
    optimizer = torch.optim.Adam(trainable, lr=1e-3)

    scaler = make_grad_scaler(device=device, enabled=True)
    loss = model._process_batch(
        batch=batch,
        is_training=True,
        optimizer=optimizer,
        scaler=scaler,
    )

    assert torch.isfinite(torch.tensor(loss))


def test_ldm_denoising_inference_loop(device, vqvae_config, ldm_config, batch):
    """
    The DDIM inference path creates tensors on ``self.device`` explicitly, so
    it is worth pinning down that it works on the selected backend.
    """
    model = LDM(
        vqvae_model_config=vqvae_config,
        ldm_model_config=ldm_config,
        device=device,
    )
    model.eval()

    config = type(
        "InferenceConfig",
        (),
        {"sample_steps": 2, "sample_root": ".", "gen_split": "gen"},
    )()

    with torch.no_grad():
        generated = model._synthesize_images_from_references(
            ref_imgs=batch["ref_img"],
            config=config,
        )

    assert len(generated) == BATCH_SIZE
    assert generated[0].size == (IMAGE_SIZE, IMAGE_SIZE)


def test_scaler_capability_matches_documented_matrix(device):
    """
    Guard the documented contract: only CUDA-like backends get a real scaler.
    """
    if device.type == "cuda":
        assert supports_grad_scaler(device) is True
    else:
        assert supports_grad_scaler(device) is False


def test_autocast_actually_engages_when_amp_is_requested(device, vqvae_config, batch):
    """
    End-to-end proof that requesting AMP really switches the forward pass to
    half precision.

    This is the regression test for the ``scaler.enabled`` bug: the models used
    to derive ``use_amp`` from an attribute that ``torch.amp.GradScaler`` does
    not have, so ``autocast`` was never entered and AMP silently did nothing.
    """
    from utils.hardware.hardware_utils import autocast_context, scaler_is_enabled

    scaler = make_grad_scaler(device=device, enabled=True)
    model = VQVAE(model_config=vqvae_config, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if not scaler_is_enabled(scaler):
        # DirectML (and CPU) fall back to float32; AMP must stay off there.
        assert device.type != "cuda"
        pytest.skip("backend has no gradient scaling; AMP is disabled by design")

    # CUDA-like backend: autocast must genuinely produce half-precision output.
    with autocast_context(device, enabled=True):
        out = model.encoder(batch["tgt_img"])
    assert out.dtype == torch.float16, (
        f"autocast did not engage: encoder produced {out.dtype}"
    )

    # And the training step must still complete and produce finite losses.
    losses = model._process_batch(
        batch=batch,
        is_training=True,
        optimizer=optimizer,
        scaler=scaler,
    )
    assert torch.isfinite(torch.tensor(losses["total"]))


class TestCheckpointPortability:
    """
    Regression tests for two bugs found while running this project on a real
    AMD GPU through DirectML.

    1. ``torch.save(model.state_dict(), ...)`` writes device tensors. On
       DirectML those pickle as ``_rebuild_device_tensor_from_numpy``, and
       reloading with the default ``weights_only=True`` fails outright.
    2. ``torch.load(file, map_location=torch.device("privateuseone"))`` crashes
       inside torch-directml, which only accepts an integer device index -- so
       *any* checkpoint load failed on that backend.
    """

    def test_save_state_dict_writes_cpu_tensors(self, device, vqvae_config):
        import torch

        model = VQVAE(model_config=vqvae_config, device=device)
        path = PROJECT_ROOT / ".scratch" / "pytest-checkpoint.pth"
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            save_state_dict(model, path)

            # The raw file must contain plain CPU tensors regardless of where
            # the model lives.
            raw = torch.load(path, map_location="cpu", weights_only=True)
            assert raw, "checkpoint is empty"
            for key, tensor in raw.items():
                assert tensor.device.type == "cpu", f"{key} is on {tensor.device}"

            # And it must load back through the project's own helper.
            restored = load_state_dict(path)
            assert set(restored) == set(model.state_dict())
            for key, tensor in restored.items():
                assert torch.equal(tensor, model.state_dict()[key].detach().cpu())
        finally:
            path.unlink(missing_ok=True)

    def test_load_state_dict_returns_cpu_tensors(self, device, vqvae_config):
        """
        ``load_state_dict`` must always hand back host tensors, never tensors
        bound to the active device.

        This matters because ``torch.load(map_location=torch.device(...))``
        crashes on DirectML (torch-directml's storage restore only accepts an
        integer index), and because a checkpoint written on one backend has to
        load on another. Keeping the load on CPU and letting the caller use
        ``model.to(device)`` sidesteps both.
        """
        import torch

        model = VQVAE(model_config=vqvae_config, device=device)
        path = PROJECT_ROOT / ".scratch" / "pytest-device-checkpoint.pth"
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            save_state_dict(model, path)

            restored = load_state_dict(path)
            assert set(restored) == set(model.state_dict())
            for key, tensor in restored.items():
                assert isinstance(tensor, torch.Tensor), key
                assert tensor.device.type == "cpu", f"{key} is on {tensor.device}"
        finally:
            path.unlink(missing_ok=True)

    def test_naively_saved_checkpoints_are_not_portable(self, device, vqvae_config):
        """
        Documents a real constraint rather than asserting a fix.

        ``torch.save(model.state_dict(), ...)`` stores the tensors as they live
        on the device. For a DirectML model those are ``privateuseone`` tensors
        which pickle as ``_rebuild_device_tensor_from_numpy``, and PyTorch's
        ``weights_only=True`` unpickler refuses that global outright -- on any
        backend, not just DirectML. This is exactly why the project saves
        through :func:`save_state_dict`, and why the test is skipped where the
        naive form cannot even be produced.
        """
        import torch

        if device.type != "cpu":
            pytest.skip(
                "device tensors cannot be round-tripped with weights_only=True; "
                "the project always saves via save_state_dict"
            )

        model = VQVAE(model_config=vqvae_config, device=device)
        path = PROJECT_ROOT / ".scratch" / "pytest-naive-checkpoint.pth"
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            torch.save(model.state_dict(), path)
            restored = load_state_dict(path)
            assert set(restored) == set(model.state_dict())
        finally:
            path.unlink(missing_ok=True)

    def test_saved_checkpoint_reloads_into_a_model_on_the_device(self, device, vqvae_config):
        """
        The full save/load cycle a user actually performs: save, build a fresh
        model, load the weights into it.
        """
        model = VQVAE(model_config=vqvae_config, device=device)
        path = PROJECT_ROOT / ".scratch" / "pytest-roundtrip.pth"
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            save_state_dict(model, path)

            fresh = VQVAE(model_config=vqvae_config, device=device)
            fresh.load_state_dict(load_state_dict(path))

            for key, tensor in model.state_dict().items():
                assert torch.equal(
                    fresh.state_dict()[key].detach().cpu(),
                    tensor.detach().cpu(),
                ), key
        finally:
            path.unlink(missing_ok=True)

    def test_load_state_dict_rejects_a_missing_file(self):
        path = PROJECT_ROOT / ".scratch" / "pytest-does-not-exist.pth"
        with pytest.raises(FileNotFoundError):
            load_state_dict(path)


class TestDeviceNameCleaning:
    """
    DirectML hands back a fixed-size wide-char buffer, so device names arrive
    padded with NUL bytes.
    """

    def test_nul_bytes_are_stripped(self):
        assert backends._clean_device_name("AMD Radeon RX 9070 XT\x00") == "AMD Radeon RX 9070 XT"

    def test_internal_nul_bytes_become_spaces(self):
        assert backends._clean_device_name("AMD\x00Radeon\x00") == "AMD Radeon"

    def test_surrounding_whitespace_is_stripped(self):
        assert backends._clean_device_name("  GPU  ") == "GPU"

    def test_a_clean_name_is_unchanged(self):
        assert backends._clean_device_name("NVIDIA GeForce RTX 4090") == "NVIDIA GeForce RTX 4090"
