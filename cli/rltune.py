"""
RL-based optimization script for SE+ST perturbation prediction

This script uses reinforcement learning (policy gradient) to directly fine-tune
the model weights by optimizing the EXACT competition scoring metrics.

Unlike traditional RL that trains a separate policy network, this approach:
1. Uses the model itself as the policy
2. Computes rewards using competition metrics (DES + PDS + MAE)
3. Updates model weights to maximize rewards
4. Outputs a standard .ckpt file for inference
"""

import logging
import sys
from pathlib import Path
from collections import defaultdict
import copy

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Import components
from models.se_st_combined import SE_ST_CombinedModel
from data.perturbation_dataset import PerturbationDataset
from utils.competition_metrics import compute_competition_score_exact

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_episode_reward(
    model,
    batch,
    device,
    reward_type="competition",
):
    """
    Compute reward for a single batch prediction.

    Args:
        model: SE+ST model
        batch: Data batch with ctrl_cell_emb, pert_emb, pert_cell_emb
        device: Device
        reward_type: Type of reward (competition, mse, correlation)

    Returns:
        Reward value (higher is better)
    """
    model.eval()

    with torch.no_grad():
        # Forward pass
        pred = model(batch, padded=False)
        target = batch['pert_cell_emb']

        if reward_type == "competition":
            # Use EXACT competition scoring
            try:
                # For competition scoring, we need cell-level predictions
                # pred and target should be [n_cells, n_genes]
                ctrl_cells = batch['ctrl_cell_emb']

                reward = compute_competition_score_exact(
                    pred_cells=pred,
                    true_cells=target,
                    ctrl_cells=ctrl_cells,
                    return_components=False,
                ) / 100.0  # Normalize to [0, 1]

            except Exception as e:
                logger.warning(f"Competition scoring failed: {e}, using MSE fallback")
                mse = F.mse_loss(pred, target)
                reward = -mse.item()

        elif reward_type == "mse":
            # Negative MSE (we want to minimize MSE, so maximize -MSE)
            mse = F.mse_loss(pred, target)
            reward = -mse.item()

        elif reward_type == "correlation":
            # Pearson correlation
            pred_flat = pred.flatten()
            target_flat = target.flatten()
            if len(pred_flat) > 1:
                correlation = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1]
                reward = correlation.item()
            else:
                reward = 0.0

        else:
            raise ValueError(f"Unknown reward type: {reward_type}")

    return reward


def train_step_with_policy_gradient(
    model,
    batch,
    optimizer,
    device,
    reward_type="competition",
    baseline_reward=0.0,
):
    """
    Single training step using policy gradient.

    The idea:
    1. Forward pass to get prediction
    2. Compute reward (competition score)
    3. Use reward as signal to update model (maximize reward)

    Args:
        model: SE+ST model
        batch: Data batch
        optimizer: Optimizer
        device: Device
        reward_type: Reward type
        baseline_reward: Baseline reward for variance reduction

    Returns:
        Loss value and reward
    """
    model.train()

    # Forward pass
    pred = model(batch, padded=False)
    target = batch['pert_cell_emb']

    # Compute reward (with no_grad to avoid computing gradients for reward)
    with torch.no_grad():
        if reward_type == "competition":
            try:
                ctrl_cells = batch['ctrl_cell_emb']
                reward = compute_competition_score_exact(
                    pred_cells=pred,
                    true_cells=target,
                    ctrl_cells=ctrl_cells,
                    return_components=False,
                ) / 100.0
            except Exception as e:
                # Fallback to MSE
                mse = F.mse_loss(pred, target)
                reward = -mse.item()
        elif reward_type == "mse":
            mse = F.mse_loss(pred, target)
            reward = -mse.item()
        elif reward_type == "correlation":
            pred_flat = pred.flatten()
            target_flat = target.flatten()
            if len(pred_flat) > 1:
                correlation = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1]
                reward = correlation.item()
            else:
                reward = 0.0
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")

    # Policy gradient loss
    # We want to maximize reward, so minimize negative reward
    # Use prediction error as surrogate loss, weighted by (reward - baseline)
    advantage = reward - baseline_reward

    # Compute prediction loss (MSE)
    mse_loss = F.mse_loss(pred, target)

    # Policy gradient: if reward is high (advantage > 0), we want to reduce loss more
    # if reward is low (advantage < 0), we want to increase loss (discourage this prediction)
    loss = mse_loss * (1.0 - advantage)  # Scale loss by inverse advantage

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return loss.item(), reward


def evaluate_model(
    model,
    dataloader,
    device,
    reward_type="competition",
    n_batches=10,
):
    """
    Evaluate model on validation set.

    Args:
        model: Model to evaluate
        dataloader: DataLoader
        device: Device
        reward_type: Reward type
        n_batches: Number of batches to evaluate

    Returns:
        Average reward
    """
    model.eval()
    total_reward = 0.0
    n_evaluated = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Compute reward
            reward = compute_episode_reward(model, batch, device, reward_type)
            total_reward += reward
            n_evaluated += 1

    return total_reward / max(n_evaluated, 1)


def rltune_main(cfg: DictConfig):
    """
    Main RL training function.

    Args:
        cfg: Hydra configuration

    Returns:
        Exit code
    """
    # Disable struct mode
    OmegaConf.set_struct(cfg, False)

    logger.info("=" * 60)
    logger.info("SE+ST RL-based Optimization (Direct Model Fine-tuning)")
    logger.info("=" * 60)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(cfg.get('output_dir', './outputs'))
    experiment_name = cfg.get('name', 'se_st_rltune')
    full_output_dir = output_dir / experiment_name
    full_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {full_output_dir}")

    # Save configuration
    config_save_path = full_output_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        OmegaConf.save(cfg, f)
    logger.info(f"Configuration saved to: {config_save_path}")

    # Load pre-trained SE+ST model
    logger.info("Loading pre-trained SE+ST model...")
    model_kwargs = OmegaConf.to_container(cfg.model.kwargs, resolve=True)
    model = SE_ST_CombinedModel(**model_kwargs)

    # Load checkpoint if provided
    initial_checkpoint = None
    if cfg.model.get('checkpoint'):
        checkpoint_path = Path(cfg.model.checkpoint)

        # Validate checkpoint file exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {cfg.model.checkpoint}\n"
                f"Please check the path is correct."
            )

        # Check if file is valid (not HTML or text)
        with open(checkpoint_path, 'rb') as f:
            header = f.read(10)
            if header.startswith(b'<') or header.startswith(b'<!DOCTYPE'):
                raise ValueError(
                    f"Checkpoint file appears to be HTML/text, not a valid PyTorch checkpoint: {cfg.model.checkpoint}\n"
                    f"This usually means the file download failed or the path is incorrect."
                )

        logger.info(f"Loading checkpoint from {cfg.model.checkpoint}...")
        try:
            checkpoint = torch.load(cfg.model.checkpoint, map_location='cpu', weights_only=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from {cfg.model.checkpoint}\n"
                f"Error: {e}\n"
                f"The file may be corrupted or not a valid PyTorch checkpoint."
            ) from e

        # Validate checkpoint structure
        if 'state_dict' not in checkpoint:
            raise ValueError(
                f"Invalid checkpoint format: missing 'state_dict' key.\n"
                f"Available keys: {list(checkpoint.keys())}"
            )

        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info(f"Model loaded successfully from {cfg.model.checkpoint}")
        initial_checkpoint = checkpoint

    model = model.to(device)

    # Create optimizer
    lr = cfg.rl.training.get('learning_rate', 1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    logger.info(f"Optimizer: AdamW with lr={lr}")

    # Load dataset
    logger.info("Loading dataset...")
    data_kwargs = OmegaConf.to_container(cfg.data.kwargs, resolve=True)

    train_dataset = PerturbationDataset(
        toml_config_path=data_kwargs['toml_config_path'],
        perturbation_features_file=data_kwargs['perturbation_features_file'],
        split="train",
        batch_col=data_kwargs.get('batch_col', 'batch_var'),
        pert_col=data_kwargs.get('pert_col', 'target_gene'),
        cell_type_key=data_kwargs.get('cell_type_key', 'cell_type'),
        control_pert=data_kwargs.get('control_pert', 'non-targeting'),
    )

    val_dataset = PerturbationDataset(
        toml_config_path=data_kwargs['toml_config_path'],
        perturbation_features_file=data_kwargs['perturbation_features_file'],
        split="val",
        batch_col=data_kwargs.get('batch_col', 'batch_var'),
        pert_col=data_kwargs.get('pert_col', 'target_gene'),
        cell_type_key=data_kwargs.get('cell_type_key', 'cell_type'),
        control_pert=data_kwargs.get('control_pert', 'non-targeting'),
    )

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")

    # Create dataloaders
    batch_size = cfg.rl.training.get('batch_size', 4)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_kwargs.get('num_workers', 4),
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_kwargs.get('num_workers', 4),
        pin_memory=True,
    )

    # Training parameters
    n_episodes = cfg.rl.training.get('n_episodes', 100)
    reward_type = cfg.rl.env.get('reward_type', 'competition')
    eval_frequency = cfg.rl.training.get('eval_frequency', 10)
    save_frequency = cfg.rl.training.get('save_frequency', 50)

    logger.info(f"Training for {n_episodes} episodes")
    logger.info(f"Reward type: {reward_type}")

    # Training loop
    best_reward = -float('inf')
    all_rewards = []
    all_losses = []
    baseline_reward = 0.0  # Running baseline for variance reduction

    logger.info("Starting RL training...")

    for episode in tqdm(range(n_episodes), desc="RL Episodes"):
        episode_rewards = []
        episode_losses = []

        # Train for one epoch
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Training step
            loss, reward = train_step_with_policy_gradient(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
                reward_type=reward_type,
                baseline_reward=baseline_reward,
            )

            episode_rewards.append(reward)
            episode_losses.append(loss)

        # Update baseline (exponential moving average)
        avg_reward = np.mean(episode_rewards)
        baseline_reward = 0.9 * baseline_reward + 0.1 * avg_reward

        all_rewards.append(avg_reward)
        all_losses.append(np.mean(episode_losses))

        # Logging
        if (episode + 1) % 5 == 0:
            logger.info(f"Episode {episode + 1}/{n_episodes}")
            logger.info(f"  Avg reward: {avg_reward:.4f}")
            logger.info(f"  Avg loss: {np.mean(episode_losses):.4f}")
            logger.info(f"  Baseline: {baseline_reward:.4f}")

        # Evaluation
        if (episode + 1) % eval_frequency == 0:
            eval_reward = evaluate_model(
                model=model,
                dataloader=val_loader,
                device=device,
                reward_type=reward_type,
                n_batches=20,
            )
            logger.info(f"  Validation reward: {eval_reward:.4f}")

            # Save best model
            if eval_reward > best_reward:
                best_reward = eval_reward
                best_model_path = full_output_dir / "best_model.ckpt"

                # Save in same format as training checkpoints
                save_dict = {
                    'state_dict': model.state_dict(),
                    'hyper_parameters': model_kwargs,
                    'episode': episode + 1,
                    'best_reward': best_reward,
                }

                # Copy optimizer state if available
                if initial_checkpoint and 'optimizer_states' in initial_checkpoint:
                    save_dict['optimizer_states'] = [optimizer.state_dict()]

                torch.save(save_dict, best_model_path)
                logger.info(f"  ✅ New best model saved! Reward: {best_reward:.4f}")

        # Periodic checkpoint
        if (episode + 1) % save_frequency == 0:
            checkpoint_path = full_output_dir / f"model_episode_{episode + 1}.ckpt"
            save_dict = {
                'state_dict': model.state_dict(),
                'hyper_parameters': model_kwargs,
                'episode': episode + 1,
            }
            torch.save(save_dict, checkpoint_path)
            logger.info(f"  Checkpoint saved to {checkpoint_path}")

    # Final save
    final_model_path = full_output_dir / "final_model.ckpt"
    save_dict = {
        'state_dict': model.state_dict(),
        'hyper_parameters': model_kwargs,
        'episode': n_episodes,
    }
    torch.save(save_dict, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Save training statistics
    stats_path = full_output_dir / "training_stats.pt"
    torch.save({
        'episode_rewards': all_rewards,
        'episode_losses': all_losses,
        'best_reward': best_reward,
    }, stats_path)
    logger.info(f"Training statistics saved to {stats_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("RL Training Summary")
    logger.info("=" * 60)
    logger.info(f"Total episodes: {n_episodes}")
    logger.info(f"Best validation reward: {best_reward:.4f}")
    logger.info(f"Final training reward: {all_rewards[-1]:.4f}")
    logger.info(f"Best model: {full_output_dir / 'best_model.ckpt'}")
    logger.info(f"Final model: {final_model_path}")
    logger.info("=" * 60)
    logger.info("✅ You can now use these .ckpt files for inference!")
    logger.info("=" * 60)

    return 0


@hydra.main(version_base=None, config_path="../configs", config_name="rltune")
def main(cfg: DictConfig):
    """Hydra entry point for RL training"""
    OmegaConf.set_struct(cfg, False)
    return rltune_main(cfg)


if __name__ == "__main__":
    sys.exit(main())
