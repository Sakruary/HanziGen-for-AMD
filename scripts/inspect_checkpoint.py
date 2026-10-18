#!/usr/bin/env python
"""
Inspect a VQ-VAE checkpoint and decide whether it is still usable.

Why this exists
---------------
``vqvae.fit()`` saves the checkpoint whenever the validation loss improves. If
training diverges, the diverged model can score a *lower* validation loss than
the healthy one (glyph images are mostly white background, so a degenerate
output still yields a small MSE). That means the file on disk can silently be
the broken model.

This script applies three independent checks, because none of them alone is
conclusive:

1. Weight sanity    -- NaN / inf / absurd magnitudes in the saved tensors.
2. Codebook usage   -- how many of the codebook entries the encoder actually
                       uses. VQ-VAE "codebook collapse" pins this to 1.
3. Reconstruction   -- real glyphs through the decoder, compared against the
                       best possible constant image. A model that cannot beat a
                       flat grey picture is useless, whatever the loss says.

Usage:
    uv run --no-sync python scripts/inspect_checkpoint.py
    uv run --no-sync python scripts/inspect_checkpoint.py ^
        --checkpoint checkpoints/vqvae_target_font.pth --samples 8
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Redirected stdout on Windows uses the console code page (GBK, CP1252, ...),
# which cannot encode emoji; force UTF-8 so the report never aborts.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - stream may not support it
            pass

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "
INFO = "ℹ️ "

# Anything whose absolute value exceeds this is treated as a numerical blow-up.
# Well-behaved conv/groupnorm weights stay far below it.
SUSPICIOUS_ABS_MAX = 1e3


def header(title: str) -> None:
    """
    Print a section header.
    """
    print(f"\n{title}")
    print("-" * len(title))


def describe(value: float) -> str:
    """
    Format a float compactly for the report.
    """
    if value == 0:
        return "0"
    if abs(value) >= 1e4 or abs(value) < 1e-3:
        return f"{value:.3e}"
    return f"{value:.4f}"


# ===== Check 1: weight sanity =====
def check_weights(state: dict[str, torch.Tensor]) -> list[str]:
    """
    Scan the checkpoint tensors for NaN, inf and blown-up magnitudes.

    Returns a list of problem descriptions (empty when everything looks sane).
    """
    header("Check 1 — weight sanity")

    problems: list[str] = []
    magnitudes: list[tuple[float, str]] = []

    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            continue
        if tensor.numel() == 0:
            continue

        nan_count = int(torch.isnan(tensor).sum())
        inf_count = int(torch.isinf(tensor).sum())

        finite = tensor[torch.isfinite(tensor)]
        abs_max = float(finite.abs().max()) if finite.numel() else float("inf")
        magnitudes.append((abs_max, name))

        if nan_count or inf_count:
            problems.append(
                f"{name}: {nan_count} NaN, {inf_count} inf (abs max "
                f"{describe(abs_max)})"
            )

    magnitudes.sort(reverse=True)

    print(f"{INFO} Tensors examined: {len(magnitudes)}")
    print(f"{INFO} Largest magnitudes:")
    for abs_max, name in magnitudes[:5]:
        flag = "  <-- suspicious" if abs_max > SUSPICIOUS_ABS_MAX else ""
        print(f"     {name:<44} abs max = {describe(abs_max)}{flag}")

    if problems:
        print(f"\n{FAIL} Found {len(problems)} tensor(s) containing NaN/inf:")
        for line in problems[:20]:
            print(f"     {line}")
        if len(problems) > 20:
            print(f"     ... and {len(problems) - 20} more")
    elif magnitudes and magnitudes[0][0] > SUSPICIOUS_ABS_MAX:
        print(
            f"\n{WARN}No NaN/inf, but weights reached {describe(magnitudes[0][0])}. "
            "That is a strong sign of divergence."
        )
    else:
        print(f"\n{PASS} All weights are finite and within a normal range.")

    return problems


# ===== Check 2 + 3: codebook usage and reconstruction =====
def load_sample_images(
    target_dir: Path,
    limit: int,
) -> tuple[torch.Tensor, list[str]]:
    """
    Load up to ``limit`` real target glyph images, normalised like training.
    """
    paths = sorted(target_dir.glob("*.png"))[:limit]
    if not paths:
        raise FileNotFoundError(f"No PNG images found in {target_dir}")

    transform = T.Compose([T.ToTensor(), T.Normalize(mean=(0.5,), std=(0.5,))])
    images = [transform(Image.open(p).convert("L")) for p in paths]

    return torch.stack(images), [p.stem for p in paths]


def quantize_indices(model, latents: torch.Tensor) -> torch.Tensor:
    """
    Reproduce the vector quantiser's nearest-neighbour lookup to expose the
    chosen codebook indices (the model's forward() does not return them).

    Computed in float32 on purpose: squaring in float16 overflows and would
    produce misleading neighbours.
    """
    flat = latents.permute(0, 2, 3, 1).reshape(-1, latents.shape[1]).float()
    embeddings = model.vector_quantizer.emb.weight.float()

    distances = (
        flat.pow(2).sum(dim=1, keepdim=True)
        + embeddings.pow(2).sum(dim=1)
        - 2 * flat @ embeddings.t()
    )

    return distances.argmin(dim=1)


def to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a normalised [-1, 1] tensor back to an 8-bit greyscale image.
    """
    array = tensor.detach().cpu().float().clamp(-1, 1)
    array = ((array + 1) / 2 * 255).round().to(torch.uint8).numpy().squeeze(0)
    return Image.fromarray(array, mode="L")


def analyse_model(
    checkpoint_state: dict[str, torch.Tensor],
    target_dir: Path,
    output_dir: Path,
    sample_count: int,
    device: torch.device,
) -> bool:
    """
    Load the checkpoint, measure codebook usage and reconstruction quality, and
    write side-by-side comparison images.

    Returns True when the checkpoint looks usable.
    """
    from configs.vqvae_config import VQVAEModelConfig  # noqa: PLC0415
    from models.vqvae.vqvae import VQVAE  # noqa: PLC0415

    model_config = VQVAEModelConfig()
    model = VQVAE(model_config=model_config, device=device)

    try:
        model.load_state_dict(checkpoint_state)
    except Exception as error:
        print(f"{FAIL} The checkpoint does not match the VQ-VAE architecture.")
        print(f"     {type(error).__name__}: {error}")
        return False

    model.eval()

    try:
        images, names = load_sample_images(target_dir, sample_count)
    except FileNotFoundError as error:
        print(f"{WARN}Could not load sample images: {error}")
        print("     Run prepare_dataset first, then re-run this script.")
        return True  # inconclusive rather than proven-bad

    images = images.to(device)

    with torch.no_grad():
        latents = model.encoder(images)
        indices = quantize_indices(model, latents)
        reconstructed, _ = model(images)

    # ---- Check 2: codebook usage ----
    header("Check 2 — codebook usage")

    used = torch.unique(indices)
    total_codes = model_config.codebook_size
    usage_ratio = len(used) / total_codes

    counts = torch.bincount(indices, minlength=total_codes)
    dominant_share = float(counts.max()) / float(counts.sum())

    print(f"{INFO} Codebook size        : {total_codes}")
    print(f"{INFO} Codes actually used  : {len(used)} ({usage_ratio:.1%})")
    print(f"{INFO} Most common code share: {dominant_share:.1%}")

    codebook_ok = True
    if len(used) == 1:
        print(f"\n{FAIL} Codebook collapse: every latent maps to a single code.")
        print("     The decoder can only ever emit one image -> useless.")
        codebook_ok = False
    elif dominant_share > 0.95:
        print(
            f"\n{FAIL} Near-total collapse: {dominant_share:.1%} of all latents "
            "use one code."
        )
        codebook_ok = False
    elif usage_ratio < 0.15:
        print(
            f"\n{WARN}Very low codebook usage ({usage_ratio:.1%}). The model is "
            "barely using its capacity."
        )
    else:
        print(f"\n{PASS} Codebook usage looks reasonable.")

    # ---- Check 3: reconstruction vs best constant image ----
    header("Check 3 — reconstruction quality")

    recon_mse = float(torch.nn.functional.mse_loss(reconstructed, images))

    # The best possible constant prediction is the image mean; any usable model
    # must beat this comfortably.
    constant_mse = float(torch.nn.functional.mse_loss(images.mean(), images))
    constant_image = torch.full_like(images, float(images.mean()))

    print(f"{INFO} Reconstruction MSE   : {recon_mse:.6f}")
    print(f"{INFO} Flat-grey baseline   : {constant_mse:.6f}")
    print(f"{INFO} (the baseline is the MSE of always outputting the average pixel)")

    recon_ok = True
    if recon_mse >= constant_mse:
        print(
            f"\n{FAIL} The model is NO BETTER than outputting a flat grey image."
        )
        print("     Whatever the loss curve said, this checkpoint carries no glyph")
        print("     information.")
        recon_ok = False
    elif recon_mse > constant_mse * 0.5:
        print(
            f"\n{WARN}Only marginally better than a flat grey image "
            f"({recon_mse / constant_mse:.0%} of the baseline)."
        )
        print("     Inspect the comparison images before trusting it.")
    else:
        improvement = (1 - recon_mse / constant_mse) * 100
        print(
            f"\n{PASS} Beats the flat-grey baseline by {improvement:.1f}%. "
            "Reconstruction carries real structure."
        )

    # ---- Write comparison images ----
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, original, rebuilt, flat in zip(
        names, images, reconstructed, constant_image
    ):
        o, r, f = to_pil(original), to_pil(rebuilt), to_pil(flat)
        width, height = o.width, o.height
        combined = Image.new("L", (width * 3, height))
        combined.paste(o, (0, 0))
        combined.paste(r, (width, 0))
        combined.paste(f, (width * 2, 0))
        combined.save(output_dir / f"{name}_orig_recon_baseline.png")

    print(f"\n{INFO} Comparison images written to: {output_dir}")
    print("     Each image is three panels: ORIGINAL | RECONSTRUCTION | FLAT GREY")
    print("     Open them: if panel 2 does not obviously resemble panel 1 (and look")
    print("     better than panel 3), the checkpoint is broken.")

    return codebook_ok and recon_ok


# ===== Entry point =====
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Check whether a VQ-VAE checkpoint is still usable",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/vqvae_target_font.pth",
        help="Path to the VQ-VAE checkpoint (default: checkpoints/vqvae_target_font.pth)",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="data/target",
        help="Directory of real target glyph images (default: data/target)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoint_check",
        help="Where to write comparison images (default: checkpoint_check)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="How many glyphs to reconstruct (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (cuda, rocm, mps, cpu). Defaults to auto-detection.",
    )
    return parser.parse_args()


def main() -> int:
    """
    Run all three checks and print a verdict.
    """
    args = parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from utils.hardware.hardware_utils import select_device  # noqa: PLC0415

        device = select_device(args.device)
    except Exception:
        device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        print(f"{FAIL} Checkpoint not found: {checkpoint_path}")
        return 1

    print(f"{INFO} Checkpoint : {checkpoint_path}")
    print(f"{INFO} Size       : {checkpoint_path.stat().st_size / 1024:.1f} KiB")
    print(f"{INFO} Written    : {datetime.fromtimestamp(checkpoint_path.stat().st_mtime)}")
    print(f"{INFO} Device     : {device}")

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        print(f"{FAIL} Could not read the checkpoint: {type(error).__name__}: {error}")
        return 1

    if not isinstance(state, dict):
        print(f"{FAIL} Unexpected checkpoint content: {type(state).__name__}")
        return 1

    weight_problems = check_weights(state)
    model_ok = analyse_model(
        checkpoint_state=state,
        target_dir=Path(args.target_dir),
        output_dir=Path(args.output_dir),
        sample_count=args.samples,
        device=device,
    )

    header("Verdict")

    if weight_problems:
        print(f"{FAIL} CORRUPTED — the saved weights contain NaN/inf.")
        print("     Delete this checkpoint and retrain.")
        return 1

    if not model_ok:
        print(f"{FAIL} NOT USABLE — this checkpoint has collapsed or diverged.")
        print("     Delete it and retrain (see the AMP / dtype notes in the docs).")
        return 1

    print(f"{PASS} No corruption detected, and the model beats a flat grey image.")
    print(
        "     Now open the comparison images and confirm by eye that panel 2\n"
        "     really reconstructs the strokes of panel 1. Numbers cannot replace\n"
        "     looking at the glyphs."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
