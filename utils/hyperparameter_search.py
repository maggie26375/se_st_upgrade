"""
Hyperparameter search utilities for AutoTune
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import json

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")


class OptunaSearcher:
    """
    Optuna-based hyperparameter search for SE+ST models.
    """

    def __init__(
        self,
        cfg: DictConfig,
        objective_fn: Callable,
        direction: str = "minimize",
        n_trials: int = 50,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
    ):
        """
        Initialize Optuna searcher.

        Args:
            cfg: Base configuration
            objective_fn: Objective function to optimize (trial -> metric)
            direction: "minimize" or "maximize"
            n_trials: Number of trials to run
            study_name: Name for the study
            storage: Database URL for distributed optimization
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter search")

        self.cfg = cfg
        self.objective_fn = objective_fn
        self.direction = direction
        self.n_trials = n_trials
        self.study_name = study_name or f"se_st_autotune_{torch.cuda.device_count()}gpu"
        self.storage = storage

        # Create study
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=direction,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1000),
            storage=storage,
            load_if_exists=True,
        )

        logger.info(f"Optuna study created: {self.study_name}")
        logger.info(f"Direction: {direction}, Trials: {n_trials}")

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters based on search space configuration.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        search_space = self.cfg.autotune.search_space
        params = {}

        for param_name, param_config in search_space.items():
            param_type = param_config.get('type')

            if param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config['choices']
                )
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    step=param_config.get('step', 1)
                )
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    step=param_config.get('step', None),
                    log=param_config.get('log', False)
                )
            elif param_type == 'uniform':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=False
                )
            elif param_type == 'loguniform':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=True
                )
            else:
                logger.warning(f"Unknown parameter type: {param_type} for {param_name}")

        logger.info(f"Trial {trial.number} suggested params: {params}")
        return params

    def update_config_with_params(self, params: Dict[str, Any]) -> DictConfig:
        """
        Update configuration with suggested hyperparameters.

        Args:
            params: Suggested hyperparameters

        Returns:
            Updated configuration
        """
        cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        OmegaConf.set_struct(cfg, False)

        # Update training parameters
        if 'lr' in params:
            cfg.training.lr = params['lr']
            cfg.model.kwargs.lr = params['lr']

        if 'batch_size' in params:
            cfg.training.batch_size = params['batch_size']

        if 'dropout' in params:
            cfg.model.kwargs.dropout = params['dropout']

        if 'gradient_clip_val' in params:
            cfg.training.gradient_clip_val = params['gradient_clip_val']

        if 'accumulate_grad_batches' in params:
            cfg.training.accumulate_grad_batches = params['accumulate_grad_batches']

        # Update model parameters
        if 'st_hidden_dim' in params:
            cfg.model.kwargs.st_hidden_dim = params['st_hidden_dim']
            # Update transformer config
            cfg.model.kwargs.transformer_backbone_kwargs.hidden_size = params['st_hidden_dim']
            cfg.model.kwargs.transformer_backbone_kwargs.intermediate_size = params['st_hidden_dim'] * 4

        if 'st_cell_set_len' in params:
            cfg.model.kwargs.st_cell_set_len = params['st_cell_set_len']
            cfg.model.kwargs.transformer_backbone_kwargs.max_position_embeddings = params['st_cell_set_len']

        if 'num_attention_heads' in params:
            num_heads = params['num_attention_heads']
            cfg.model.kwargs.transformer_backbone_kwargs.num_attention_heads = num_heads
            cfg.model.kwargs.transformer_backbone_kwargs.num_key_value_heads = num_heads
            # Update head_dim
            hidden_dim = cfg.model.kwargs.st_hidden_dim
            cfg.model.kwargs.transformer_backbone_kwargs.head_dim = hidden_dim // num_heads

        if 'num_hidden_layers' in params:
            cfg.model.kwargs.transformer_backbone_kwargs.num_hidden_layers = params['num_hidden_layers']
            cfg.model.kwargs.n_encoder_layers = params['num_hidden_layers']
            cfg.model.kwargs.n_decoder_layers = params['num_hidden_layers']

        if 'distributional_loss' in params:
            cfg.model.kwargs.distributional_loss = params['distributional_loss']

        if 'blur' in params:
            cfg.model.kwargs.blur = params['blur']

        return cfg

    def optimize(self) -> optuna.Study:
        """
        Run hyperparameter optimization.

        Returns:
            Completed Optuna study
        """
        logger.info("Starting hyperparameter optimization...")

        def objective_wrapper(trial):
            # Suggest hyperparameters
            params = self.suggest_hyperparameters(trial)

            # Update config
            trial_cfg = self.update_config_with_params(params)

            # Run objective function
            metric = self.objective_fn(trial, trial_cfg)

            return metric

        # Run optimization
        self.study.optimize(
            objective_wrapper,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        # Log results
        logger.info("Optimization completed!")
        logger.info(f"Best trial: {self.study.best_trial.number}")
        logger.info(f"Best value: {self.study.best_value}")
        logger.info(f"Best params: {self.study.best_params}")

        return self.study

    def save_results(self, output_dir: Path):
        """
        Save optimization results.

        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save best parameters
        best_params_path = output_dir / "best_params.json"
        with open(best_params_path, 'w') as f:
            json.dump(self.study.best_params, f, indent=2)
        logger.info(f"Best parameters saved to {best_params_path}")

        # Save study summary
        summary_path = output_dir / "study_summary.json"
        summary = {
            'best_trial': self.study.best_trial.number,
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'n_trials': len(self.study.trials),
            'direction': self.direction,
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Study summary saved to {summary_path}")

        # Save all trials
        trials_path = output_dir / "all_trials.json"
        trials_data = []
        for trial in self.study.trials:
            trials_data.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state),
            })
        with open(trials_path, 'w') as f:
            json.dump(trials_data, f, indent=2)
        logger.info(f"All trials saved to {trials_path}")

        # Try to save visualization if plotly is available
        try:
            import optuna.visualization as vis

            # Optimization history
            fig = vis.plot_optimization_history(self.study)
            fig.write_html(str(output_dir / "optimization_history.html"))

            # Parameter importances
            try:
                fig = vis.plot_param_importances(self.study)
                fig.write_html(str(output_dir / "param_importances.html"))
            except:
                logger.warning("Could not generate parameter importance plot")

            # Parallel coordinate plot
            try:
                fig = vis.plot_parallel_coordinate(self.study)
                fig.write_html(str(output_dir / "parallel_coordinate.html"))
            except:
                logger.warning("Could not generate parallel coordinate plot")

            logger.info("Visualization plots saved")
        except ImportError:
            logger.warning("Plotly not available for visualization. Install with: pip install plotly")


class TrialPruningCallback:
    """
    Callback for pruning unpromising trials during training.
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "val_loss"):
        """
        Initialize pruning callback.

        Args:
            trial: Optuna trial object
            monitor: Metric to monitor for pruning
        """
        self.trial = trial
        self.monitor = monitor

    def __call__(self, trainer, pl_module):
        """
        Check if trial should be pruned.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
        """
        # Get current step
        current_step = trainer.global_step

        # Get current metric
        if self.monitor in trainer.callback_metrics:
            current_value = trainer.callback_metrics[self.monitor].item()

            # Report to Optuna
            self.trial.report(current_value, current_step)

            # Check if should prune
            if self.trial.should_prune():
                raise optuna.TrialPruned(f"Trial pruned at step {current_step}")


def create_best_config(base_cfg: DictConfig, best_params: Dict[str, Any]) -> DictConfig:
    """
    Create final configuration with best hyperparameters.

    Args:
        base_cfg: Base configuration
        best_params: Best hyperparameters from optimization

    Returns:
        Configuration with best hyperparameters
    """
    searcher = OptunaSearcher(
        cfg=base_cfg,
        objective_fn=lambda x, y: 0,  # Dummy function
    )
    best_cfg = searcher.update_config_with_params(best_params)

    logger.info("Best configuration created:")
    logger.info(f"\n{OmegaConf.to_yaml(best_cfg)}")

    return best_cfg
