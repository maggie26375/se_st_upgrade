"""
Reward models for RL-based perturbation prediction optimization
"""

import logging
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """
    Learned reward model for perturbation predictions.

    Trained to predict the quality of predictions based on features.
    """

    def __init__(
        self,
        input_dim: int = 1024,  # prediction + target features
        hidden_dims: list = [512, 256, 128],
        activation: str = "relu",
        dropout: float = 0.1,
    ):
        """
        Initialize reward model.

        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim

        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (scalar reward)
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

        logger.info(f"Reward model initialized with {sum(p.numel() for p in self.parameters())} parameters")

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict reward for a prediction.

        Args:
            prediction: Model prediction [B, D]
            target: Ground truth [B, D]
            features: Optional additional features [B, F]

        Returns:
            reward: Predicted reward [B, 1]
        """
        # Compute basic metrics as features
        diff = prediction - target
        abs_diff = torch.abs(diff)
        squared_diff = diff ** 2

        # Cosine similarity
        cosine_sim = F.cosine_similarity(prediction, target, dim=-1, eps=1e-8).unsqueeze(-1)

        # Pearson correlation (approximation)
        pred_mean = prediction.mean(dim=-1, keepdim=True)
        target_mean = target.mean(dim=-1, keepdim=True)
        pred_centered = prediction - pred_mean
        target_centered = target - target_mean
        correlation = (pred_centered * target_centered).mean(dim=-1, keepdim=True)

        # Aggregate statistics
        mean_abs_error = abs_diff.mean(dim=-1, keepdim=True)
        mean_squared_error = squared_diff.mean(dim=-1, keepdim=True)
        max_error = abs_diff.max(dim=-1, keepdim=True)[0]

        # Concatenate all features
        feature_list = [
            prediction.mean(dim=-1, keepdim=True),  # Pred mean
            target.mean(dim=-1, keepdim=True),  # Target mean
            prediction.std(dim=-1, keepdim=True),  # Pred std
            target.std(dim=-1, keepdim=True),  # Target std
            cosine_sim,
            correlation,
            mean_abs_error,
            mean_squared_error,
            max_error,
        ]

        if features is not None:
            feature_list.append(features)

        # If input_dim is large, we might need raw prediction/target
        if self.input_dim > 100:
            # Include sampled or aggregated pred/target
            feature_list.extend([prediction, target])

        input_features = torch.cat(feature_list, dim=-1)

        # Ensure correct dimension
        if input_features.shape[-1] < self.input_dim:
            # Pad with zeros
            padding = torch.zeros(
                input_features.shape[0],
                self.input_dim - input_features.shape[-1],
                device=input_features.device
            )
            input_features = torch.cat([input_features, padding], dim=-1)
        elif input_features.shape[-1] > self.input_dim:
            # Truncate
            input_features = input_features[:, :self.input_dim]

        # Forward through network
        reward = self.net(input_features)

        return reward


class PairwiseRewardModel(nn.Module):
    """
    Pairwise reward model for ranking predictions.

    Learns to compare two predictions and determine which is better.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: list = [256, 128],
        activation: str = "relu",
    ):
        """
        Initialize pairwise reward model.

        Args:
            input_dim: Input dimension per prediction
            hidden_dims: Hidden layer dimensions
            activation: Activation function
        """
        super().__init__()

        # Feature extractor for each prediction
        feature_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            feature_layers.append(nn.ReLU() if activation == "relu" else nn.GELU())
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*feature_layers)

        # Comparison head
        self.comparison_head = nn.Sequential(
            nn.Linear(prev_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Probability that first is better
        )

    def forward(
        self,
        pred1: torch.Tensor,
        pred2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compare two predictions.

        Args:
            pred1: First prediction [B, D]
            pred2: Second prediction [B, D]

        Returns:
            preference: Probability that pred1 is better than pred2 [B, 1]
        """
        # Extract features
        feat1 = self.feature_extractor(pred1)
        feat2 = self.feature_extractor(pred2)

        # Concatenate and compare
        combined = torch.cat([feat1, feat2], dim=-1)
        preference = self.comparison_head(combined)

        return preference


def train_reward_model(
    reward_model: RewardModel,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ground_truth_rewards: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    n_epochs: int = 100,
    batch_size: int = 32,
) -> float:
    """
    Train reward model using supervised learning.

    Args:
        reward_model: Reward model to train
        predictions: Model predictions [N, D]
        targets: Ground truth targets [N, D]
        ground_truth_rewards: True rewards [N]
        optimizer: Optimizer
        n_epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        Final loss
    """
    reward_model.train()
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle data
        indices = torch.randperm(len(predictions))

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]

            pred_batch = predictions[batch_indices]
            target_batch = targets[batch_indices]
            reward_batch = ground_truth_rewards[batch_indices].unsqueeze(-1)

            # Forward
            predicted_rewards = reward_model(pred_batch, target_batch)

            # Loss
            loss = criterion(predicted_rewards, reward_batch)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        if (epoch + 1) % 10 == 0:
            logger.info(f"Reward model training epoch {epoch + 1}/{n_epochs}, loss: {avg_loss:.4f}")

    return avg_loss


def compute_intrinsic_reward(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reward_type: str = "mse",
) -> float:
    """
    Compute intrinsic reward without learned model.

    Args:
        prediction: Model prediction
        target: Ground truth
        reward_type: Type of reward

    Returns:
        reward value
    """
    with torch.no_grad():
        if reward_type == "mse":
            return -F.mse_loss(prediction, target).item()
        elif reward_type == "mae":
            return -F.l1_loss(prediction, target).item()
        elif reward_type == "cosine":
            return F.cosine_similarity(prediction.flatten(), target.flatten(), dim=0).item()
        elif reward_type == "correlation":
            pred_flat = prediction.flatten()
            target_flat = target.flatten()
            corr_matrix = torch.corrcoef(torch.stack([pred_flat, target_flat]))
            return corr_matrix[0, 1].item()
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")
