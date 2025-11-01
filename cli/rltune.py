"""
RL-based optimization script for SE+ST perturbation prediction
"""

import logging
import sys
from pathlib import Path
from collections import defaultdict

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from tqdm import tqdm

# Import components
from se_st_upgrade.models.se_st_combined import SE_ST_CombinedModel
from se_st_upgrade.models.rl_agent import PPOAgent
from se_st_upgrade.utils.rl_environment import create_rl_env
from se_st_upgrade.data.perturbation_dataset import PerturbationDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RolloutBuffer:
    """Buffer for storing rollout trajectories."""

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, obs, action, reward, value, log_prob, done):
        """Add transition to buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get(self):
        """Get all data as tensors."""
        return {
            'observations': torch.stack(self.observations),
            'actions': torch.stack(self.actions),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32),
            'values': torch.stack(self.values).squeeze(-1),
            'log_probs': torch.stack(self.log_probs).squeeze(-1),
            'dones': torch.tensor(self.dones, dtype=torch.float32),
        }

    def clear(self):
        """Clear buffer."""
        self.__init__()

    def __len__(self):
        return len(self.observations)


def collect_rollout(
    env,
    agent: PPOAgent,
    buffer: RolloutBuffer,
    n_steps: int,
) -> dict:
    """
    Collect rollout data.

    Args:
        env: RL environment
        agent: RL agent
        buffer: Rollout buffer
        n_steps: Number of steps to collect

    Returns:
        Episode statistics
    """
    stats = defaultdict(list)
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0

    for step in range(n_steps):
        # Select action
        obs_tensor = torch.from_numpy(obs).float()
        action, log_prob, value = agent.select_action(obs_tensor)

        # Take step
        next_obs, reward, done, truncated, info = env.step(action.cpu().numpy())
        done = done or truncated

        # Store transition
        buffer.add(
            obs=obs_tensor,
            action=action,
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=done,
        )

        # Update statistics
        episode_reward += reward
        episode_length += 1

        # Reset if done
        if done:
            stats['episode_reward'].append(episode_reward)
            stats['episode_length'].append(episode_length)
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
        else:
            obs = next_obs

    return dict(stats)


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
    logger.info("SE+ST RL-based Optimization")
    logger.info("=" * 60)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

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

    if cfg.model.get('checkpoint'):
        checkpoint = torch.load(cfg.model.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info(f"Model loaded from {cfg.model.checkpoint}")

    model.eval()  # Set to eval mode for RL

    # Load dataset
    logger.info("Loading dataset...")
    data_kwargs = OmegaConf.to_container(cfg.data.kwargs, resolve=True)
    dataset = PerturbationDataset(
        toml_config_path=data_kwargs['toml_config_path'],
        perturbation_features_file=data_kwargs['perturbation_features_file'],
        split="train",
        batch_col=data_kwargs.get('batch_col', 'batch_var'),
        pert_col=data_kwargs.get('pert_col', 'target_gene'),
        cell_type_key=data_kwargs.get('cell_type_key', 'cell_type'),
        control_pert=data_kwargs.get('control_pert', 'non-targeting'),
    )
    logger.info(f"Dataset loaded with {len(dataset)} samples")

    # Create RL environment
    logger.info("Creating RL environment...")
    env_kwargs = OmegaConf.to_container(cfg.rl.env, resolve=True)
    env = create_rl_env(model, dataset, **env_kwargs)
    logger.info("RL environment created")

    # Create RL agent
    logger.info("Creating RL agent...")
    agent_kwargs = OmegaConf.to_container(cfg.rl.agent, resolve=True)
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        **agent_kwargs,
    )
    logger.info("RL agent created")

    # Training loop
    logger.info("Starting RL training...")
    n_episodes = cfg.rl.training.n_episodes
    rollout_buffer_size = cfg.rl.training.rollout_buffer_size
    eval_frequency = cfg.rl.training.eval_frequency
    save_frequency = cfg.rl.training.save_frequency

    buffer = RolloutBuffer()
    global_step = 0
    best_reward = -float('inf')

    # Training statistics
    all_episode_rewards = []
    all_episode_lengths = []

    for episode in tqdm(range(n_episodes), desc="RL Training"):
        # Collect rollout
        stats = collect_rollout(env, agent, buffer, rollout_buffer_size)

        # Update agent
        if len(buffer) >= rollout_buffer_size:
            rollout_data = buffer.get()
            train_stats = agent.update(
                rollout_data,
                n_epochs=cfg.rl.training.n_epochs_per_update,
                batch_size=cfg.rl.training.batch_size,
            )

            # Log training statistics
            logger.info(f"Episode {episode + 1}/{n_episodes}")
            logger.info(f"  Actor loss: {train_stats['actor_loss']:.4f}")
            logger.info(f"  Critic loss: {train_stats['critic_loss']:.4f}")

            if stats['episode_reward']:
                avg_reward = np.mean(stats['episode_reward'])
                avg_length = np.mean(stats['episode_length'])
                logger.info(f"  Avg episode reward: {avg_reward:.4f}")
                logger.info(f"  Avg episode length: {avg_length:.1f}")

                all_episode_rewards.extend(stats['episode_reward'])
                all_episode_lengths.extend(stats['episode_length'])

                # Save best agent
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_agent_path = full_output_dir / "best_agent.pt"
                    agent.save(str(best_agent_path))
                    logger.info(f"  New best reward: {best_reward:.4f}, agent saved")

            # Clear buffer
            buffer.clear()
            global_step += rollout_buffer_size

        # Periodic evaluation
        if (episode + 1) % eval_frequency == 0:
            logger.info(f"Evaluation at episode {episode + 1}")
            eval_reward = evaluate_agent(env, agent, n_episodes=5)
            logger.info(f"  Evaluation reward: {eval_reward:.4f}")

        # Periodic save
        if (episode + 1) % save_frequency == 0:
            checkpoint_path = full_output_dir / f"agent_episode_{episode + 1}.pt"
            agent.save(str(checkpoint_path))
            logger.info(f"  Agent saved to {checkpoint_path}")

    # Final save
    final_agent_path = full_output_dir / "final_agent.pt"
    agent.save(str(final_agent_path))
    logger.info(f"Final agent saved to {final_agent_path}")

    # Save training statistics
    stats_path = full_output_dir / "training_stats.pt"
    torch.save({
        'episode_rewards': all_episode_rewards,
        'episode_lengths': all_episode_lengths,
        'best_reward': best_reward,
    }, stats_path)
    logger.info(f"Training statistics saved to {stats_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("RL Training Summary")
    logger.info("=" * 60)
    logger.info(f"Total episodes: {n_episodes}")
    logger.info(f"Best average reward: {best_reward:.4f}")
    if all_episode_rewards:
        logger.info(f"Final 100-episode avg reward: {np.mean(all_episode_rewards[-100:]):.4f}")
    logger.info(f"Agent saved to: {final_agent_path}")
    logger.info("=" * 60)

    return 0


def evaluate_agent(env, agent: PPOAgent, n_episodes: int = 10) -> float:
    """
    Evaluate agent performance.

    Args:
        env: Environment
        agent: RL agent
        n_episodes: Number of evaluation episodes

    Returns:
        Average reward
    """
    total_reward = 0.0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            obs_tensor = torch.from_numpy(obs).float()
            action, _, _ = agent.select_action(obs_tensor, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action.cpu().numpy())
            done = done or truncated
            episode_reward += reward

        total_reward += episode_reward

    return total_reward / n_episodes


@hydra.main(version_base=None, config_path="../configs", config_name="rltune")
def main(cfg: DictConfig):
    """Hydra entry point for RL training"""
    OmegaConf.set_struct(cfg, False)
    return rltune_main(cfg)


if __name__ == "__main__":
    sys.exit(main())
