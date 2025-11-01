"""
Reinforcement Learning Environment for Perturbation Prediction Optimization
"""

import logging
from typing import Dict, Tuple, Optional, List
import numpy as np

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PerturbationPredictionEnv:
    """
    RL Environment for optimizing perturbation prediction strategies.

    The agent learns to:
    1. Select optimal control cells for comparison
    2. Adjust prediction confidence/strength
    3. Optimize cell sampling strategies
    """

    def __init__(
        self,
        model: nn.Module,
        dataset,
        state_dim: int = 512,
        num_control_cells: int = 128,
        max_episode_steps: int = 50,
        reward_type: str = "negative_mse",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize RL environment.

        Args:
            model: Pre-trained SE+ST model
            dataset: Dataset for evaluation
            state_dim: Dimension of state embeddings
            num_control_cells: Number of control cells to select
            max_episode_steps: Maximum steps per episode
            reward_type: Type of reward (negative_mse, negative_energy, correlation)
            device: Device to run on
        """
        self.model = model
        self.dataset = dataset
        self.state_dim = state_dim
        self.num_control_cells = num_control_cells
        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type
        self.device = device

        # Move model to device
        self.model.to(device)
        self.model.eval()

        # Environment state
        self.current_step = 0
        self.current_perturbation = None
        self.current_control_pool = None
        self.current_target = None
        self.selected_controls = []

        # Action space: [select_control_idx, confidence_weight]
        # select_control_idx: which control cell to add (0 to pool_size-1)
        # confidence_weight: how much to trust this prediction (0.0 to 1.0)
        self.action_dim = 2

        # Observation space: [perturbation_embedding, current_state_embedding, step_info]
        self.obs_dim = state_dim * 2 + 10  # perturbation + avg_control_state + metadata

        logger.info(f"RL Environment initialized:")
        logger.info(f"  State dim: {state_dim}")
        logger.info(f"  Action dim: {self.action_dim}")
        logger.info(f"  Observation dim: {self.obs_dim}")

    def reset(self) -> np.ndarray:
        """
        Reset environment for new episode.

        Returns:
            Initial observation
        """
        self.current_step = 0
        self.selected_controls = []

        # Sample a random perturbation from dataset
        sample = self.dataset[np.random.randint(len(self.dataset))]

        # Extract components
        self.current_perturbation = sample['pert_emb'].to(self.device)
        self.current_control_pool = sample['ctrl_cell_emb'].to(self.device)
        self.current_target = sample['pert_cell_emb'].to(self.device)

        # Initial observation
        obs = self._get_observation()

        return obs

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.

        Returns:
            Observation array
        """
        with torch.no_grad():
            # Perturbation embedding
            pert_emb = self.current_perturbation.cpu().numpy()

            # Average control state (so far)
            if len(self.selected_controls) > 0:
                selected_indices = [idx for idx, _ in self.selected_controls]
                avg_control = self.current_control_pool[selected_indices].mean(dim=0).cpu().numpy()
            else:
                avg_control = np.zeros(self.state_dim)

            # Metadata
            metadata = np.array([
                self.current_step / self.max_episode_steps,  # Progress
                len(self.selected_controls) / self.num_control_cells,  # Selection ratio
                len(self.selected_controls),  # Num selected
                self.num_control_cells - len(self.selected_controls),  # Remaining
                0.0,  # Current reward (filled in later)
                0.0, 0.0, 0.0, 0.0, 0.0,  # Reserved
            ])

            # Concatenate
            obs = np.concatenate([pert_emb[:self.state_dim], avg_control, metadata])

        return obs.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: [select_control_idx, confidence_weight]

        Returns:
            (observation, reward, done, info)
        """
        self.current_step += 1

        # Parse action
        select_idx = int(action[0] * len(self.current_control_pool)) % len(self.current_control_pool)
        confidence = float(np.clip(action[1], 0.0, 1.0))

        # Add to selected controls
        self.selected_controls.append((select_idx, confidence))

        # Check if episode is done
        done = (
            self.current_step >= self.max_episode_steps or
            len(self.selected_controls) >= self.num_control_cells
        )

        # Compute reward
        reward = self._compute_reward() if done else 0.0

        # Get new observation
        obs = self._get_observation()

        # Info
        info = {
            'step': self.current_step,
            'num_selected': len(self.selected_controls),
            'final_reward': reward if done else None,
        }

        return obs, reward, done, info

    def _compute_reward(self) -> float:
        """
        Compute reward for current state.

        Uses EXACT competition scoring if reward_type='competition',
        otherwise uses simpler proxy rewards.

        Returns:
            Reward value
        """
        if len(self.selected_controls) == 0:
            return -1.0  # Penalty for no selection

        with torch.no_grad():
            # Get selected control cells and weights
            selected_indices = [idx for idx, _ in self.selected_controls]
            confidence_weights = torch.tensor(
                [conf for _, conf in self.selected_controls],
                device=self.device
            )

            # Weighted average of selected controls
            selected_controls = self.current_control_pool[selected_indices]
            weights = confidence_weights / confidence_weights.sum()
            weighted_control = (selected_controls * weights.unsqueeze(1)).sum(dim=0, keepdim=True)

            # Create batch for model
            batch = {
                'ctrl_cell_emb': weighted_control,
                'pert_emb': self.current_perturbation.unsqueeze(0),
                'pert_cell_emb': self.current_target.unsqueeze(0),
            }

            # Forward through model
            prediction = self.model.forward(batch, padded=False)

            # Compute reward based on type
            target = self.current_target.unsqueeze(0)

            if self.reward_type == "competition":
                # Use EXACT competition scoring (DES + PDS + MAE)
                try:
                    from .competition_metrics import compute_competition_score_exact

                    # Note: prediction and target should be single cells or cell sets
                    # For now we compute on single prediction, ideally would batch
                    reward = compute_competition_score_exact(
                        pred_cells=prediction,
                        true_cells=target,
                        ctrl_cells=weighted_control,
                        # all_true_pseudobulks and perturbation_idx would need dataset context
                        return_components=False,
                    ) / 100.0  # Normalize to [0, 1]
                except Exception as e:
                    logger.warning(f"Competition scoring failed: {e}, using MSE fallback")
                    mse = torch.nn.functional.mse_loss(prediction, target)
                    reward = -mse.item()

            elif self.reward_type == "negative_mse":
                # Negative MSE (higher is better)
                mse = torch.nn.functional.mse_loss(prediction, target)
                reward = -mse.item()

            elif self.reward_type == "negative_energy":
                # Negative energy distance
                try:
                    from .energy_loss import energy_distance
                    energy = energy_distance(prediction, target)
                    reward = -energy.item()
                except:
                    logger.warning("Energy distance not available, using MSE")
                    mse = torch.nn.functional.mse_loss(prediction, target)
                    reward = -mse.item()

            elif self.reward_type == "correlation":
                # Pearson correlation
                pred_flat = prediction.flatten()
                target_flat = target.flatten()
                correlation = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1]
                reward = correlation.item()

            else:
                raise ValueError(f"Unknown reward type: {self.reward_type}")

        return reward

    def render(self, mode='human'):
        """Render environment (for debugging)."""
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.max_episode_steps}")
            print(f"Selected controls: {len(self.selected_controls)}/{self.num_control_cells}")
            if self.selected_controls:
                indices, confidences = zip(*self.selected_controls)
                print(f"  Indices: {indices}")
                print(f"  Confidences: {[f'{c:.3f}' for c in confidences]}")


class PerturbationEnvWrapper:
    """
    Gym-like wrapper for RL frameworks (e.g., Stable Baselines3).
    """

    def __init__(self, env: PerturbationPredictionEnv):
        self.env = env
        self.observation_space = self._create_obs_space()
        self.action_space = self._create_action_space()

    def _create_obs_space(self):
        """Create observation space."""
        try:
            from gymnasium import spaces
            return spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.env.obs_dim,),
                dtype=np.float32
            )
        except ImportError:
            logger.warning("gymnasium not installed, observation_space not created")
            return None

    def _create_action_space(self):
        """Create action space."""
        try:
            from gymnasium import spaces
            return spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.env.action_dim,),
                dtype=np.float32
            )
        except ImportError:
            logger.warning("gymnasium not installed, action_space not created")
            return None

    def reset(self):
        """Reset environment."""
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        """Take step."""
        obs, reward, done, info = self.env.step(action)
        truncated = False
        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        """Render."""
        return self.env.render(mode)

    def close(self):
        """Close environment."""
        pass


def create_rl_env(model, dataset, **kwargs) -> PerturbationEnvWrapper:
    """
    Create RL environment for perturbation prediction.

    Args:
        model: SE+ST model
        dataset: Dataset
        **kwargs: Additional arguments

    Returns:
        Wrapped environment
    """
    env = PerturbationPredictionEnv(model, dataset, **kwargs)
    wrapped_env = PerturbationEnvWrapper(env)
    logger.info("RL environment created")
    return wrapped_env
