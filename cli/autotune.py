"""
AutoTune script for SE+ST model hyperparameter optimization
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Import training utilities
from cli.train import SE_ST_DataModule, setup_logger
from models.se_st_combined import SE_ST_CombinedModel
from utils.hyperparameter_search import OptunaSearcher, TrialPruningCallback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_trial_objective(cfg: DictConfig):
    """
    Create objective function for Optuna optimization.

    Args:
        cfg: Base configuration

    Returns:
        Objective function that takes (trial, trial_cfg) and returns metric
    """

    def objective(trial, trial_cfg):
        """
        Objective function for a single trial.

        Args:
            trial: Optuna trial object
            trial_cfg: Configuration for this trial

        Returns:
            Validation loss (metric to minimize)
        """
        try:
            # Disable struct mode
            OmegaConf.set_struct(trial_cfg, False)

            # Create unique experiment name for this trial
            trial_name = f"{trial_cfg.name}_trial_{trial.number}"
            trial_output_dir = Path(trial_cfg.output_dir) / trial_name
            trial_output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Starting trial {trial.number}")
            logger.info(f"Trial output: {trial_output_dir}")

            # Save trial config
            config_path = trial_output_dir / "trial_config.yaml"
            with open(config_path, 'w') as f:
                OmegaConf.save(trial_cfg, f)

            # Initialize model
            logger.info("Initializing model...")
            model_kwargs = OmegaConf.to_container(
                trial_cfg.model.kwargs, resolve=True
            )
            model = SE_ST_CombinedModel(**model_kwargs)

            # Initialize data module
            logger.info("Initializing data module...")
            data_kwargs = OmegaConf.to_container(
                trial_cfg.data.kwargs, resolve=True
            )
            data_kwargs['batch_size'] = trial_cfg.training.batch_size
            datamodule = SE_ST_DataModule(**data_kwargs)

            # Setup callbacks
            callbacks = []

            # Model checkpoint
            checkpoint_callback = ModelCheckpoint(
                dirpath=trial_output_dir / "checkpoints",
                filename="best",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=False,
            )
            callbacks.append(checkpoint_callback)

            # Early stopping
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=trial_cfg.autotune.trial_early_stopping.patience,
                min_delta=trial_cfg.autotune.trial_early_stopping.min_delta,
                verbose=True,
            )
            callbacks.append(early_stop_callback)

            # Pruning callback
            pruning_callback = TrialPruningCallback(trial, monitor="val_loss")

            # Setup trainer
            trainer = Trainer(
                max_steps=trial_cfg.autotune.resources.max_steps,
                accelerator=trial_cfg.get('accelerator', 'auto'),
                devices=trial_cfg.autotune.resources.devices,
                callbacks=callbacks,
                logger=False,  # Disable logger for trials
                enable_checkpointing=True,
                log_every_n_steps=100,
                val_check_interval=trial_cfg.autotune.resources.val_check_interval,
                precision=trial_cfg.autotune.resources.precision,
                gradient_clip_val=trial_cfg.training.gradient_clip_val,
                accumulate_grad_batches=trial_cfg.training.accumulate_grad_batches,
            )

            # Train model
            logger.info(f"Training trial {trial.number}...")
            trainer.fit(model, datamodule)

            # Get best validation loss
            best_val_loss = checkpoint_callback.best_model_score.item()

            logger.info(f"Trial {trial.number} completed with val_loss={best_val_loss:.4f}")

            return best_val_loss

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            logger.exception("Full traceback:")
            # Return a large value to indicate failure
            return float('inf')

    return objective


def autotune_main(cfg: DictConfig):
    """
    Main AutoTune function.

    Args:
        cfg: Hydra configuration

    Returns:
        Exit code
    """
    # Disable struct mode
    OmegaConf.set_struct(cfg, False)

    logger.info("=" * 60)
    logger.info("SE+ST AutoTune - Hyperparameter Optimization")
    logger.info("=" * 60)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create output directory
    output_dir = Path(cfg.output_dir) / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Save base configuration
    config_path = output_dir / "base_config.yaml"
    with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f)
    logger.info(f"Base configuration saved to {config_path}")

    # Create objective function
    objective_fn = create_trial_objective(cfg)

    # Initialize Optuna searcher
    logger.info("Initializing Optuna searcher...")
    searcher = OptunaSearcher(
        cfg=cfg,
        objective_fn=objective_fn,
        direction=cfg.autotune.study.direction,
        n_trials=cfg.autotune.n_trials,
        study_name=cfg.name,
    )

    # Run optimization
    logger.info(f"Starting optimization with {cfg.autotune.n_trials} trials...")
    study = searcher.optimize()

    # Save results
    logger.info("Saving optimization results...")
    searcher.save_results(output_dir)

    # Create best configuration
    logger.info("Creating best configuration...")
    best_cfg = searcher.update_config_with_params(study.best_params)

    # Save best configuration
    best_config_path = output_dir / "best_config.yaml"
    with open(best_config_path, 'w') as f:
        OmegaConf.save(best_cfg, f)
    logger.info(f"Best configuration saved to {best_config_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("AutoTune Optimization Summary")
    logger.info("=" * 60)
    logger.info(f"Total trials: {len(study.trials)}")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best validation loss: {study.best_value:.4f}")
    logger.info(f"Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)

    logger.info(f"\nTo train with best hyperparameters, run:")
    logger.info(f"  se-st-train --config-path {output_dir} --config-name best_config")

    return 0


@hydra.main(version_base=None, config_path="../configs", config_name="autotune")
def main(cfg: DictConfig):
    """Hydra entry point for AutoTune"""
    OmegaConf.set_struct(cfg, False)
    return autotune_main(cfg)


if __name__ == "__main__":
    sys.exit(main())
