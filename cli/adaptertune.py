"""
AdapterTune script for parameter-efficient fine-tuning of SE+ST models
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Import training utilities
from se_st_upgrade.cli.train import SE_ST_DataModule, setup_logger, setup_callbacks
from se_st_upgrade.models.adapter_se_st import AdapterSE_ST_Model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdapterCheckpoint(ModelCheckpoint):
    """
    Custom checkpoint callback that saves only adapter weights.
    """

    def __init__(self, save_adapters_only: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.save_adapters_only = save_adapters_only

    def _save_checkpoint(self, trainer, filepath):
        """Save checkpoint with optional adapter-only mode."""
        if self.save_adapters_only:
            # Save only adapter weights
            model = trainer.model
            if isinstance(model, AdapterSE_ST_Model):
                model.save_adapters(filepath.replace('.ckpt', '_adapters.pt'))
                logger.info(f"Saved adapter weights to {filepath}")
            else:
                logger.warning("Model is not AdapterSE_ST_Model, saving full checkpoint")
                super()._save_checkpoint(trainer, filepath)
        else:
            # Save full model
            super()._save_checkpoint(trainer, filepath)


def adaptertune_main(cfg: DictConfig):
    """
    Main AdapterTune function.

    Args:
        cfg: Hydra configuration

    Returns:
        Exit code
    """
    # Disable struct mode
    OmegaConf.set_struct(cfg, False)

    logger.info("=" * 60)
    logger.info("SE+ST AdapterTune - Parameter-Efficient Fine-Tuning")
    logger.info("=" * 60)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create output directory
    output_dir = Path(cfg.get('output_dir', './outputs'))
    experiment_name = cfg.get('name', 'se_st_adaptertune')
    full_output_dir = output_dir / experiment_name
    full_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {full_output_dir}")

    # Save configuration
    config_save_path = full_output_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        OmegaConf.save(cfg, f)
    logger.info(f"Configuration saved to: {config_save_path}")

    # Setup loggers
    loggers = setup_logger(cfg)

    # Setup callbacks
    callbacks = []

    # Adapter checkpoint callback
    adapter_save_path = full_output_dir / "adapters"
    adapter_save_path.mkdir(parents=True, exist_ok=True)

    if cfg.get('adapter', {}).get('save_adapters_only', True):
        checkpoint_callback = AdapterCheckpoint(
            save_adapters_only=True,
            dirpath=adapter_save_path,
            filename="step={step}-val_loss={val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=cfg.get('save_top_k', 3),
            save_last=False,
            verbose=True,
            every_n_train_steps=cfg.training.get('ckpt_every_n_steps', 2000),
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=full_output_dir / "checkpoints",
            filename="step={step}-val_loss={val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=cfg.get('save_top_k', 3),
            save_last=False,
            verbose=True,
            every_n_train_steps=cfg.training.get('ckpt_every_n_steps', 2000),
        )

    callbacks.append(checkpoint_callback)

    # Early stopping
    if cfg.training.get('early_stopping', False):
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=cfg.training.get('patience', 5),
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    # Initialize trainer
    trainer = Trainer(
        max_steps=cfg.training.get('max_steps', 10000),
        max_epochs=cfg.training.get('max_epochs', None),
        accelerator=cfg.get('accelerator', 'auto'),
        devices=cfg.get('devices', 'auto'),
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=cfg.training.get('log_every_n_steps', 50),
        val_check_interval=cfg.training.get('val_check_interval', 500),
        precision=cfg.training.get('precision', '16-mixed' if torch.cuda.is_available() else '32'),
        gradient_clip_val=cfg.training.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=cfg.training.get('accumulate_grad_batches', 1),
    )

    logger.info("Trainer initialized")
    logger.info(f"Device: {cfg.get('accelerator', 'auto')}")

    # Initialize model
    logger.info("Initializing AdapterSE_ST_Model...")
    model_kwargs = OmegaConf.to_container(cfg.get('model', {}).get('kwargs', {}), resolve=True)

    model = AdapterSE_ST_Model(**model_kwargs)
    logger.info(f"Model initialized with {cfg.model.kwargs.adapter_type} adapters")

    # Load pre-trained base model if specified
    if cfg.model.get('checkpoint'):
        logger.info(f"Loading base model from {cfg.model.checkpoint}")
        checkpoint = torch.load(cfg.model.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("Base model loaded successfully")

    # Initialize data module
    logger.info("Initializing data module...")
    data_kwargs = OmegaConf.to_container(cfg.get('data', {}).get('kwargs', {}), resolve=True)

    if 'batch_size' not in data_kwargs:
        data_kwargs['batch_size'] = cfg.training.get('batch_size', 32)

    datamodule = SE_ST_DataModule(**data_kwargs)
    logger.info("Data module initialized")

    # Log parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("=" * 60)
    logger.info("Model Parameter Statistics")
    logger.info("=" * 60)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    logger.info(f"Memory savings: {100 * (1 - trainable_params / total_params):.2f}%")
    logger.info("=" * 60)

    # Start training
    logger.info("Starting adapter fine-tuning...")
    logger.info(f"Max steps: {cfg.training.get('max_steps', 10000)}")
    logger.info(f"Batch size: {data_kwargs.get('batch_size', 32)}")

    try:
        trainer.fit(model, datamodule)
        logger.info("Adapter fine-tuning completed successfully!")

        # Save final adapters
        final_adapter_path = full_output_dir / "final_adapters.pt"
        model.save_adapters(str(final_adapter_path))
        logger.info(f"Final adapters saved to {final_adapter_path}")

        # Save full model if requested
        if not cfg.get('adapter', {}).get('save_adapters_only', True):
            final_model_path = full_output_dir / "final_model.ckpt"
            trainer.save_checkpoint(str(final_model_path))
            logger.info(f"Final model saved to {final_model_path}")

        logger.info("=" * 60)
        logger.info("AdapterTune Summary")
        logger.info("=" * 60)
        logger.info(f"Adapter type: {cfg.model.kwargs.adapter_type}")
        logger.info(f"Total training steps: {trainer.global_step}")
        logger.info(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
        logger.info(f"Adapter weights: {final_adapter_path}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Adapter fine-tuning failed: {e}")
        logger.exception("Full traceback:")
        return 1


@hydra.main(version_base=None, config_path="../configs", config_name="adaptertune")
def main(cfg: DictConfig):
    """Hydra entry point for AdapterTune"""
    OmegaConf.set_struct(cfg, False)
    return adaptertune_main(cfg)


if __name__ == "__main__":
    sys.exit(main())
