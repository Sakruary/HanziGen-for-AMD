from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from rich.console import Console
from rich.table import Table
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.rich import tqdm

from configs.vqvae_config import VQVAEModelConfig, VQVAETrainingConfig
from datasets.loader import Loader
from utils.hardware.hardware_utils import (
    autocast_context,
    save_state_dict,
    scaler_is_enabled,
    select_device,
)
from utils.image.image_utils import convert_tensor_to_pil_images

from .vqvae_encoder_decoder import VQVAEDecoder, VQVAEEncoder, VQVAEQuantizer


class TrainingDivergedError(RuntimeError):
    """
    Raised when the VQ-VAE has stopped carrying any information about the glyph.

    This is deliberately loud: a collapsed model can keep reporting a low loss
    for hours (glyph images are mostly white background), so the run must stop
    rather than silently waste the GPU.
    """


class VQVAE(nn.Module):
    """
    Vector Quantized Variational Autoencoder (VQ-VAE).
    """

    # ===== Initialization & Setup =====
    def __init__(
        self,
        model_config: VQVAEModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()

        # Initialize model device
        self.device = select_device(device)

        # Define encoder, vector quantizer, and decoder
        self.encoder = VQVAEEncoder(
            in_channels=model_config.input_img_channels,
            base_channels=model_config.encoder_base_channels,
            out_channels=model_config.latent_dim,
        )
        self.vector_quantizer = VQVAEQuantizer(
            codebook_size=model_config.codebook_size,
            latent_dim=model_config.latent_dim,
            commitment_cost=model_config.commitment_cost,
        )
        self.decoder = VQVAEDecoder(
            in_channels=model_config.latent_dim,
            base_channels=model_config.encoder_base_channels,
            out_channels=model_config.input_img_channels,
        )

        # Define loss function
        self.loss_fn = nn.MSELoss()

        # Move model to device
        self.to(self.device)

    # ===== Core Operations =====
    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the VQ-VAE model.
        """

        encoded_features = self.encoder(x)
        quantized_features, vq_loss = self.vector_quantizer(encoded_features)
        x_recon = self.decoder(quantized_features)

        return x_recon, vq_loss

    def _process_batch(
        self,
        batch: dict[str, Tensor],
        is_training: bool,
        optimizer: Optimizer | None = None,
        scaler=None,
    ) -> dict[str, float]:
        """
        Process a single batch of data.

        ``scaler`` is a :class:`torch.amp.GradScaler` on CUDA-like backends and
        a no-op fallback elsewhere. Mixed precision itself comes from
        :func:`~utils.hardware.hardware_utils.autocast_context`, which is a
        context manager on every backend.

        ``use_amp`` and ``use_scaler`` are kept separate on purpose: bfloat16
        runs with autocast enabled but with gradient scaling disabled (it needs
        none), while float16 runs with both enabled. Deriving autocast from
        ``scaler_is_enabled`` would silently switch autocast off for bf16.
        """
        use_amp = scaler is not None
        use_scaler = scaler_is_enabled(scaler)

        tgt_imgs = batch["tgt_img"].to(self.device)
        ref_imgs = batch["ref_img"].to(self.device)

        with autocast_context(self.device, enabled=use_amp):
            tgt_x_recon, tgt_vq_loss = self(tgt_imgs)
            tgt_recon_loss = self.loss_fn(tgt_x_recon, tgt_imgs)

            ref_x_recon, ref_vq_loss = self(ref_imgs)
            ref_recon_loss = self.loss_fn(ref_x_recon, ref_imgs)

            recon_loss = (tgt_recon_loss + ref_recon_loss) / 2
            vq_loss = (tgt_vq_loss + ref_vq_loss) / 2
            total_loss = recon_loss + vq_loss

        if is_training and optimizer is not None:
            optimizer.zero_grad()
            if use_scaler:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

        losses = {
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "vq": vq_loss.item(),
        }

        return losses

    # ===== Training and Validation =====
    def fit(
        self,
        loader: Loader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        training_config: VQVAETrainingConfig,
        scaler=None,
    ) -> None:
        """
        Train the VQ-VAE model and save the best model checkpoint.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        base_sample_root = Path(training_config.sample_root)
        training_config.sample_root = str(
            base_sample_root / f"vqvae_training_{timestamp}"
        )
        print(f"📁 Training samples will be saved to: {training_config.sample_root}")

        log_dir = Path(training_config.tensorboard_log_dir) / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"📊 TensorBoard logs will be saved to: {log_dir}")

        with SummaryWriter(log_dir) as writer:
            model_save_path = Path(training_config.model_save_path)
            model_save_path.parent.mkdir(parents=True, exist_ok=True)

            min_val_loss = float("inf")
            consecutive_unhealthy = 0
            health_patience = 2

            for epoch in range(training_config.num_epochs):

                # Train and validate for one epoch
                train_losses, val_losses = self._run_epoch(
                    loader=loader,
                    optimizer=optimizer,
                    scaler=scaler,
                )

                # Update learning rate
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]

                # Log training and validation metrics
                self._log_training_metrics(
                    writer=writer,
                    epoch=epoch,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    learning_rate=current_lr,
                )

                # Health check BEFORE any checkpoint is written: a collapsed
                # model must never be stored as the "best" one.
                healthy, health_summary = self._check_training_health(
                    val_loader=loader.loader.val,
                )
                print(f"🩺 Health: {health_summary}")

                if healthy:
                    consecutive_unhealthy = 0
                else:
                    consecutive_unhealthy += 1
                    print(
                        f"❌ The model is no longer reconstructing glyph "
                        f"structure ({consecutive_unhealthy}/{health_patience} "
                        "consecutive checks failed)."
                    )
                    if consecutive_unhealthy >= health_patience:
                        raise TrainingDivergedError(
                            "VQ-VAE training has collapsed: the model is no "
                            "better than a flat grey image and/or the codebook "
                            "has collapsed. Stopping instead of wasting more "
                            f"GPU time. Last health check: {health_summary}"
                        )

                self._save_epoch_visualization_samples(
                    loader=loader,
                    training_config=training_config,
                    epoch=epoch,
                )

                # Save the best model checkpoint, gated on `healthy` on purpose:
                # a collapsed model can post a lower validation loss than a good
                # one, and saving it would silently destroy the last usable
                # weights.
                if healthy and val_losses["total"] < min_val_loss:
                    min_val_loss = val_losses["total"]
                    save_state_dict(self, model_save_path)
                    print(f"✅ Best model saved (val loss: {min_val_loss:.6f})")
                elif not healthy:
                    print(
                        "⏭️  Checkpoint NOT saved this epoch: the health check "
                        "failed (see above)."
                    )

                # Print Metrics
                self._print_epoch_status(
                    epoch=epoch + 1,
                    total_epochs=training_config.num_epochs,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    learning_rate=current_lr,
                )

    def _run_epoch(
        self,
        loader: Loader,
        optimizer: Optimizer,
        scaler=None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Run one epoch of training and validation.
        """
        train_loader = loader.loader.train
        val_loader = loader.loader.val

        train_losses = self._train_one_epoch(
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_losses = self._validate_one_epoch(
            val_loader=val_loader,
            scaler=scaler,
        )

        return train_losses, val_losses

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: Optimizer,
        scaler=None,
    ) -> dict[str, float]:
        """
        Train the model for one epoch.
        """
        self.train()
        epoch_losses = {"total": 0.0, "recon": 0.0, "vq": 0.0}

        for batch in tqdm(train_loader, desc="Training"):
            batch_losses = self._process_batch(
                batch=batch,
                is_training=True,
                optimizer=optimizer,
                scaler=scaler,
            )
            for k in epoch_losses.keys():
                epoch_losses[k] += batch_losses[k]

        num_batches = len(train_loader)
        return {
            loss_type: loss / num_batches for loss_type, loss in epoch_losses.items()
        }

    @torch.no_grad()
    def _validate_one_epoch(
        self,
        val_loader: DataLoader,
        scaler=None,
    ) -> dict[str, float]:
        """
        Validate the model for one epoch.
        """
        self.eval()
        epoch_losses = {"total": 0.0, "recon": 0.0, "vq": 0.0}

        for batch in tqdm(val_loader, desc="Validating"):
            batch_losses = self._process_batch(
                batch=batch,
                is_training=False,
                optimizer=None,
                scaler=scaler,
            )
            for k in epoch_losses.keys():
                epoch_losses[k] += batch_losses[k]

        num_batches = len(val_loader)
        return {
            loss_type: loss / num_batches for loss_type, loss in epoch_losses.items()
        }

    @torch.no_grad()
    def _check_training_health(
        self,
        val_loader: DataLoader,
        max_mse_ratio: float = 1.0,
        max_codebook_share: float = 0.95,
    ) -> tuple[bool, str]:
        """
        Decide whether the model still reconstructs real glyph structure.

        The training loss alone cannot answer this. Glyph images are mostly
        white background, so a collapsed model that emits near-constant output
        still scores a small MSE -- which is exactly how a dead run can look
        healthy for hours. Two signals separate the cases:

        * reconstruction MSE versus the flat-grey baseline (the MSE of always
          outputting the average pixel). A usable model must beat that baseline;
        * how much of the codebook the encoder actually uses. VQ-VAE collapse
          pins this to one or two entries.

        Returns ``(healthy, human readable summary)``.
        """
        try:
            batch = next(iter(val_loader))
        except StopIteration:
            return True, "skipped (empty validation loader)"

        images = batch["tgt_img"].to(self.device)

        reconstructed, _ = self(images)

        recon_mse = torch.nn.functional.mse_loss(reconstructed, images).item()
        baseline = torch.nn.functional.mse_loss(images.mean(), images).item()
        mse_ratio = recon_mse / baseline if baseline > 0 else float("inf")

        # Count the codebook entries the encoder actually uses.
        latents = self.encoder(images)
        flat = latents.permute(0, 2, 3, 1).reshape(-1, latents.shape[1]).float()
        embeddings = self.vector_quantizer.emb.weight.float()
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            + embeddings.pow(2).sum(dim=1)
            - 2 * flat @ embeddings.t()
        )
        indices = distances.argmin(dim=1)
        used_codes = int(torch.unique(indices).numel())
        codebook_size = int(self.vector_quantizer.codebook_size)
        dominant_share = float(torch.bincount(indices).max()) / float(indices.numel())

        summary = (
            f"recon MSE {recon_mse:.6f} vs flat-grey {baseline:.6f} "
            f"(ratio {mse_ratio:.3f}) | codebook {used_codes}/{codebook_size} used, "
            f"top code {dominant_share:.1%}"
        )

        healthy = mse_ratio < max_mse_ratio and dominant_share <= max_codebook_share

        return healthy, summary

    @torch.no_grad()
    def _save_epoch_visualization_samples(
        self,
        loader: Loader,
        training_config: VQVAETrainingConfig,
        epoch: int,
    ) -> None:
        """
        Generate and save reconstruction sample images from training and validation loaders.
        """
        self.eval()

        if (
            epoch % training_config.img_save_interval != 0
            and epoch != training_config.num_epochs - 1
        ):
            return

        train_loader = loader.loader.train
        val_loader = loader.loader.val

        train_split = training_config.train_split
        val_split = training_config.val_split

        for data_split, data_loader in tqdm(
            [(train_split, train_loader), (val_split, val_loader)],
            desc="Generating samples images",
        ):
            self._generate_double_comparison_images(
                loader=data_loader,
                training_config=training_config,
                epoch=epoch,
                split=data_split,
            )

    # ===== Image Generation =====
    @torch.no_grad()
    def _generate_double_comparison_images(
        self,
        loader: DataLoader,
        training_config: VQVAETrainingConfig,
        epoch: int,
        split: str,
    ) -> None:
        """
        Generate double comparison images: Original | Reconstructed
        """
        self.eval()

        sample_dir = Path(training_config.sample_root) / split
        sample_dir.mkdir(parents=True, exist_ok=True)

        batch = next(iter(loader))
        tgt_imgs = batch["tgt_img"].to(self.device)
        ref_imgs = batch["ref_img"].to(self.device)
        img_names = batch["img_name"]

        tgt_recon, _ = self(tgt_imgs)
        ref_recon, _ = self(ref_imgs)

        for orig_tgt, recon_tgt, orig_ref, recon_ref, img_name in zip(
            tgt_imgs, tgt_recon, ref_imgs, ref_recon, img_names
        ):
            tgt_combined = self._create_combined_image(orig_tgt, recon_tgt)
            tgt_img_path = sample_dir / f"epoch_{epoch:04d}_tgt_{img_name}.png"
            tgt_combined.save(tgt_img_path)

            ref_combined = self._create_combined_image(orig_ref, recon_ref)
            ref_img_path = sample_dir / f"epoch_{epoch:04d}_ref_{img_name}.png"
            ref_combined.save(ref_img_path)

    def _create_combined_image(
        self,
        orig: Tensor,
        recon: Tensor,
    ) -> Image.Image:
        """
        Create a double comparison image: original | reconstructed.
        """

        orig_pil_img = convert_tensor_to_pil_images(orig)
        recon_pil_img = convert_tensor_to_pil_images(recon)

        width, height = orig_pil_img.width, orig_pil_img.height
        combined_img = Image.new("L", (width * 2, height))

        combined_img.paste(orig_pil_img, (0, 0))
        combined_img.paste(recon_pil_img, (width, 0))

        return combined_img

    # ===== Logging =====
    def _log_training_metrics(
        self,
        writer: SummaryWriter,
        epoch: int,
        train_losses: dict,
        val_losses: dict,
        learning_rate: float,
    ) -> None:
        """
        Log training and validation metrics to TensorBoard.
        """
        for loss_type, loss_value in train_losses.items():
            writer.add_scalar(f"Loss/train/{loss_type}", loss_value, epoch)

        for loss_type, loss_value in val_losses.items():
            writer.add_scalar(f"Loss/val/{loss_type}", loss_value, epoch)

        writer.add_scalar("LR", learning_rate, epoch)

    def _print_epoch_status(
        self,
        epoch: int,
        total_epochs: int,
        train_losses: dict,
        val_losses: dict,
        learning_rate: float,
    ) -> None:
        """
        Print the status of the current epoch using a rich table.
        """
        console = Console()
        table = Table(title=f"Epoch Status [{epoch}/{total_epochs}]")

        table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
        table.add_column("Train Loss", justify="right", style="magenta")
        table.add_column("Val Loss", justify="right", style="green")

        for loss_type in train_losses.keys():
            table.add_row(
                loss_type.capitalize(),
                f"{train_losses[loss_type]:.6f}",
                f"{val_losses[loss_type]:.6f}",
            )

        table.add_row("Learning Rate", f"{learning_rate:.6f}", "-")

        console.print(table)
