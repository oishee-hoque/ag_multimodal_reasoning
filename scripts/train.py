"""
Hydra-powered training entry point.

Usage:
    # Step 1: RGB baseline
    python scripts/train.py experiment=step1_rgb_baseline

    # Step 2: with noise reduction
    python scripts/train.py experiment=step2_noise_reduced

    # Override from command line
    python scripts/train.py data.band_group=spectral_s4 model.name=unet_resnet50

    # On SLURM
    sbatch slurm_job.sh scripts/train.py experiment=step1_rgb_baseline
"""

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from irrigation.data.datamodule import IrrigationDataModule
from irrigation.modules.seg_module import SegmentationModule
from irrigation.data.bands import get_band_config


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    # Seed
    pl.seed_everything(cfg.seed, workers=True)

    # Data
    band_config = get_band_config(cfg.data.band_group)
    datamodule = IrrigationDataModule(
        train_state_path=cfg.paths.train_state,
        test_state_path=cfg.paths.get("test_state"),
        band_group=cfg.data.band_group,
        split_mode=cfg.data.split_mode,
        val_fraction=cfg.data.val_fraction,
        noise_strategy=cfg.data.noise_strategy,
        high_threshold=cfg.data.get("high_threshold", 0.4),
        low_threshold=cfg.data.get("low_threshold", 0.15),
        ndvi_threshold=cfg.data.get("ndvi_threshold", 0.4),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        seed=cfg.seed,
    )

    # Model
    module = SegmentationModule(
        model_name=cfg.model.name,
        in_channels=band_config.num_channels,
        num_classes=cfg.model.num_classes,
        class_weights=cfg.model.get("class_weights"),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        scheduler=cfg.training.scheduler,
        max_epochs=cfg.training.max_epochs,
    )

    # Logger
    logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.get("experiment_name", None),
        tags=cfg.get("tags", []),
        log_model=True,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val/mIoU",
            mode="max",
            save_top_k=3,
            filename="{epoch}-{val/mIoU:.4f}",
        ),
        EarlyStopping(
            monitor="val/mIoU",
            mode="max",
            patience=cfg.training.get("patience", 20),
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu",
        devices=cfg.training.get("gpus", 1),
        strategy=cfg.training.get("strategy", "auto"),
        precision=cfg.training.get("precision", "16-mixed"),
        logger=logger,
        callbacks=callbacks,
        deterministic=cfg.get("deterministic", False),
        log_every_n_steps=cfg.training.get("log_every_n_steps", 10),
    )

    # Train
    trainer.fit(module, datamodule=datamodule)

    # Test on best checkpoint
    trainer.test(module, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    train()
