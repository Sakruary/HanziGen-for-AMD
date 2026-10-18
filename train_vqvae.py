import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs import VQVAEDatasetConfig, VQVAEModelConfig, VQVAETrainingConfig
from datasets.loader import Loader
from models import TrainingDivergedError, VQVAE
from utils.argparse.argparse_utils import update_config_from_args
from utils.hardware.hardware_utils import (
    enable_fast_math,
    load_state_dict,
    make_grad_scaler,
    print_backend_banner,
    print_model_params,
    resolve_amp_settings,
    select_device,
)
from utils.validation.project_validator import ProjectValidationError, ProjectValidator


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for VQ-VAE training.
    """
    parser = argparse.ArgumentParser(
        description="Train VQVAE model",
    )
    parser.add_argument(
        "--split_ratios",
        type=float,
        nargs=2,
        help="Train/val split ratios",
    )
    parser.add_argument(
        "--split_random_seed",
        type=int,
        help="Split random seed",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        help="Model save path",
    )
    parser.add_argument(
        "--tensorboard_log_dir",
        type=str,
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--sample_root",
        type=str,
        help="Sample root directory",
    )
    parser.add_argument(
        "--img_save_interval",
        type=int,
        help="Image save interval",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Compute device: auto (default), cuda, rocm/hip/amd, zluda, "
            "directml, mps, cpu (an index such as cuda:0 is also accepted)"
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="false",
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--use_amp",
        type=str,
        default="false",
        help=(
            "Use Automatic Mixed Precision (true/false). Silently falls back to "
            "float32 on backends without GradScaler support (for example DirectML)"
        ),
    )
    return parser.parse_args()


def load_vqvae_checkpoint(
    vqvae: VQVAE,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """
    Load VQ-VAE model weights from checkpoint.
    """
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"❌ VQ-VAE model checkpoint not found: {checkpoint_path}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"❌ VQ-VAE model checkpoint is not a file: {checkpoint_path}"
        )

    print(f"📁 Loading VQ-VAE model checkpoint: {checkpoint_path}")
    state_dict = load_state_dict(checkpoint)
    vqvae.load_state_dict(state_dict)


def train_vqvae(
    dataset_config: VQVAEDatasetConfig,
    model_config: VQVAEModelConfig,
    training_config: VQVAETrainingConfig,
    device: torch.device,
    resume: bool = False,
    use_amp: bool = False,
):
    """
    Train VQ-VAE model.
    """
    enable_fast_math(device)

    loader = Loader.from_dataset_config(
        dataset_config=dataset_config,
        device=device,
    )

    vqvae = VQVAE(
        model_config=model_config,
        device=device,
    )

    optimizer = optim.Adam(
        vqvae.parameters(),
        lr=training_config.learning_rate,
    )

    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=training_config.num_epochs,
        eta_min=training_config.min_learning_rate,
    )

    scaler = make_grad_scaler(device=device, enabled=use_amp) if use_amp else None

    if resume:
        load_vqvae_checkpoint(
            vqvae=vqvae,
            checkpoint_path=training_config.model_save_path,
            device=device,
        )

    print_model_params(
        model=vqvae,
    )

    vqvae.fit(
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config=training_config,
        scaler=scaler,
    )


def main() -> None:
    """
    Main function to run the VQ-VAE training process.
    """
    try:
        args = parse_args()
        resume = args.resume.lower() == "true"
        use_amp_requested = args.use_amp.lower() == "true"
        device = select_device(args.device)
        use_amp, amp_warning = resolve_amp_settings(device, use_amp_requested)
        if amp_warning:
            print(amp_warning)
        print_backend_banner(device=device, use_amp=use_amp)

        if resume:
            ProjectValidator.validate_checkpoint_file(
                file_path=args.model_save_path,
                name="VQ-VAE model checkpoint",
            )

        dataset_config = update_config_from_args(
            converting_config=VQVAEDatasetConfig(),
            args=args,
        )
        model_config = update_config_from_args(
            converting_config=VQVAEModelConfig(),
            args=args,
        )
        training_config = update_config_from_args(
            converting_config=VQVAETrainingConfig(),
            args=args,
        )

        train_vqvae(
            dataset_config=dataset_config,
            model_config=model_config,
            training_config=training_config,
            device=device,
            resume=resume,
            use_amp=use_amp,
        )

        print("✅ VQ-VAE training completed successfully")

    except TrainingDivergedError as e:
        print(f"❌ {e}")
        print(
            "❌ The saved checkpoint is NOT usable and should be deleted before\n"
            "   retraining. Common causes: float16 overflow in automatic mixed\n"
            "   precision, or a learning rate that is too high. Try\n"
            "   HANZIGEN_AMP_DTYPE=bf16, or USE_AMP=false."
        )
        sys.exit(1)

    except ProjectValidationError as e:
        print(f"❌ {e}")
        print("❌ VQ-VAE training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
