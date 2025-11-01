"""
RL Agent and Policy Networks for Perturbation Prediction Optimization
"""

import logging
from typing import Tuple, Optional, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """
    Actor network for RL agent.

    Outputs actions for cell selection and confidence weighting.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        activation: str = "relu",
        log_std_init: float = 0.0,
    ):
        """
        Initialize policy network.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            log_std_init: Initial log standard deviation
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network layers
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim

        self.shared_net = nn.Sequential(*layers)

        # Mean head
        self.mean_head = nn.Linear(prev_dim, action_dim)

        # Log std head (learnable)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
        nn.init.constant_(self.log_std_head.weight, 0.0)
        nn.init.constant_(self.log_std_head.bias, log_std_init)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get action.

        Args:
            obs: Observation tensor [B, obs_dim]
            deterministic: Whether to sample deterministically

        Returns:
            (action, log_prob)
        """
        # Shared features
        features = self.shared_net(obs)

        # Mean and std
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, -20, 2)  # Stability
        std = torch.exp(log_std)

        if deterministic:
            # Deterministic action
            action = torch.tanh(mean)
            log_prob = None
        else:
            # Sample from Gaussian
            dist = Normal(mean, std)
            sample = dist.rsample()  # Reparameterization trick
            action = torch.tanh(sample)

            # Log probability with tanh correction
            log_prob = dist.log_prob(sample)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action without log prob."""
        action, _ = self.forward(obs, deterministic)
        return action


class ValueNetwork(nn.Module):
    """
    Critic network for RL agent.

    Estimates state value V(s) or Q(s, a).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: Optional[int] = None,
        hidden_dims: list = [256, 256],
        activation: str = "relu",
        network_type: str = "v",  # "v" for V(s), "q" for Q(s,a)
    ):
        """
        Initialize value network.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension (only for Q-network)
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            network_type: "v" or "q"
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.network_type = network_type

        # Input dimension
        if network_type == "q":
            assert action_dim is not None, "action_dim required for Q-network"
            input_dim = obs_dim + action_dim
        else:
            input_dim = obs_dim

        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim

        # Output layer (single value)
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass to get value.

        Args:
            obs: Observation tensor [B, obs_dim]
            action: Action tensor [B, action_dim] (only for Q-network)

        Returns:
            value: Value tensor [B, 1]
        """
        if self.network_type == "q":
            assert action is not None, "Action required for Q-network"
            x = torch.cat([obs, action], dim=-1)
        else:
            x = obs

        value = self.net(x)
        return value


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_clip: float = 0.2,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize PPO agent.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_ratio: PPO clip ratio
            value_clip: Value function clip ratio
            entropy_coef: Entropy coefficient
            max_grad_norm: Max gradient norm
            device: Device
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Networks
        self.actor = PolicyNetwork(obs_dim, action_dim).to(device)
        self.critic = ValueNetwork(obs_dim, network_type="v").to(device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        logger.info(f"PPO Agent initialized on {device}")

    def select_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action given observation.

        Args:
            obs: Observation [obs_dim]
            deterministic: Whether to act deterministically

        Returns:
            (action, log_prob, value)
        """
        obs = obs.to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            action, log_prob = self.actor(obs, deterministic)
            value = self.critic(obs)

        return action.squeeze(0), log_prob.squeeze(0) if log_prob is not None else None, value.squeeze(0)

    def update(
        self,
        rollout_buffer,
        n_epochs: int = 10,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """
        Update policy using PPO.

        Args:
            rollout_buffer: Buffer with rollout data
            n_epochs: Number of update epochs
            batch_size: Mini-batch size

        Returns:
            Training statistics
        """
        # Compute advantages
        advantages, returns = self._compute_gae(
            rollout_buffer['rewards'],
            rollout_buffer['values'],
            rollout_buffer['dones'],
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training statistics
        stats = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy': 0.0,
            'kl_div': 0.0,
        }

        # Update for n_epochs
        for epoch in range(n_epochs):
            # Sample mini-batches
            indices = torch.randperm(len(rollout_buffer['observations']))

            for start_idx in range(0, len(indices), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]

                # Get batch data
                obs_batch = rollout_buffer['observations'][batch_indices].to(self.device)
                actions_batch = rollout_buffer['actions'][batch_indices].to(self.device)
                old_log_probs_batch = rollout_buffer['log_probs'][batch_indices].to(self.device)
                advantages_batch = advantages[batch_indices].to(self.device)
                returns_batch = returns[batch_indices].to(self.device)
                old_values_batch = rollout_buffer['values'][batch_indices].to(self.device)

                # Actor loss
                _, log_probs = self.actor(obs_batch)
                ratio = torch.exp(log_probs - old_log_probs_batch)

                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_batch
                actor_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                # entropy = -log_probs.mean()  # Simplified
                # actor_loss -= self.entropy_coef * entropy

                # Critic loss
                values = self.critic(obs_batch)
                if self.value_clip:
                    values_clipped = old_values_batch + torch.clamp(
                        values - old_values_batch,
                        -self.value_clip,
                        self.value_clip
                    )
                    critic_loss1 = F.mse_loss(values, returns_batch)
                    critic_loss2 = F.mse_loss(values_clipped, returns_batch)
                    critic_loss = torch.max(critic_loss1, critic_loss2)
                else:
                    critic_loss = F.mse_loss(values, returns_batch)

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # Update stats
                stats['actor_loss'] += actor_loss.item()
                stats['critic_loss'] += critic_loss.item()

        # Average stats
        n_updates = n_epochs * math.ceil(len(indices) / batch_size)
        for key in stats:
            stats[key] /= n_updates

        return stats

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Rewards [T]
            values: Value estimates [T]
            dones: Done flags [T]

        Returns:
            (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values

        return advantages, returns

    def save(self, path: str):
        """Save agent."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
        logger.info(f"Agent saved to {path}")

    def load(self, path: str):
        """Load agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        logger.info(f"Agent loaded from {path}")
