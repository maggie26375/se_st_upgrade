"""
Adapter modules for parameter-efficient fine-tuning of SE+ST models.

Implements:
1. Bottleneck Adapters (Houlsby et al., 2019)
2. LoRA (Low-Rank Adaptation) (Hu et al., 2021)
3. Prefix Tuning (Li & Liang, 2021)
"""

import logging
from typing import Optional, Dict, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BottleneckAdapter(nn.Module):
    """
    Bottleneck adapter module for parameter-efficient fine-tuning.

    Architecture:
        Input -> Down-projection -> Activation -> Up-projection -> Add residual
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 64,
        activation: str = "gelu",
        dropout: float = 0.1,
        init_scale: float = 1e-3,
    ):
        """
        Initialize bottleneck adapter.

        Args:
            input_dim: Input/output dimension
            bottleneck_dim: Bottleneck dimension (typically much smaller than input_dim)
            activation: Activation function (gelu, relu, silu)
            dropout: Dropout rate
            init_scale: Initialization scale for weights
        """
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        # Down-projection
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)

        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Up-projection
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize with small weights (near-identity at initialization)
        nn.init.normal_(self.down_proj.weight, std=init_scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=init_scale)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor [B, L, D]

        Returns:
            Output tensor [B, L, D]
        """
        residual = x

        # Adapter transformation
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)

        # Residual connection
        return residual + x


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) for linear layers.

    Replaces W with W + BA, where B and A are low-rank matrices.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        """
        Initialize LoRA linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Rank of low-rank decomposition
            alpha: Scaling factor
            dropout: Dropout rate
            merge_weights: Whether to merge LoRA weights into original weights
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Original linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.linear.weight.requires_grad = False
        self.linear.bias.requires_grad = False

        # LoRA parameters (trainable)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Merge weights flag
        self.merge_weights = merge_weights
        self.merged = False

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        if self.merged:
            # Use merged weights
            return self.linear(x)
        else:
            # Compute W*x + (B*A)*x
            result = self.linear(x)
            lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
            lora_out = self.dropout(lora_out)
            return result + lora_out

    def merge_lora_weights(self):
        """Merge LoRA weights into the original linear layer."""
        if not self.merged:
            # W_new = W + alpha/r * B*A
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True
            logger.info(f"LoRA weights merged for {self}")

    def unmerge_lora_weights(self):
        """Unmerge LoRA weights from the original linear layer."""
        if self.merged:
            # W_new = W - alpha/r * B*A
            self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
            logger.info(f"LoRA weights unmerged for {self}")


class PrefixTuning(nn.Module):
    """
    Prefix tuning for transformer models.

    Prepends trainable prefix embeddings to the input sequence.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        prefix_length: int = 20,
        prefix_projection: bool = True,
        projection_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize prefix tuning.

        Args:
            num_layers: Number of transformer layers
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            prefix_length: Length of prefix sequence
            prefix_projection: Whether to use MLP reparameterization
            projection_dim: Dimension for MLP projection
            dropout: Dropout rate
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.prefix_length = prefix_length
        self.head_dim = hidden_dim // num_heads

        if prefix_projection:
            # Use MLP to reparameterize prefix (more stable training)
            self.projection_dim = projection_dim or hidden_dim // 2

            # Prefix embeddings (to be projected)
            self.prefix_embed = nn.Parameter(
                torch.randn(num_layers, 2, prefix_length, self.projection_dim)
            )

            # MLP projection
            self.prefix_mlp = nn.Sequential(
                nn.Linear(self.projection_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 2 * hidden_dim),  # For key and value
            )

        else:
            # Direct prefix embeddings
            self.prefix_embed = nn.Parameter(
                torch.randn(num_layers, 2, prefix_length, hidden_dim)
            )
            self.prefix_mlp = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.normal_(self.prefix_embed, std=0.02)

    def forward(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get prefix key and value for a specific layer.

        Args:
            layer_idx: Transformer layer index

        Returns:
            Tuple of (prefix_key, prefix_value), each [batch_size, num_heads, prefix_length, head_dim]
        """
        # Get prefix embeddings for this layer
        prefix = self.prefix_embed[layer_idx]  # [2, prefix_length, dim]

        # Project if using MLP reparameterization
        if self.prefix_mlp is not None:
            prefix = self.prefix_mlp(prefix)  # [2, prefix_length, 2*hidden_dim]
            prefix = prefix.view(2, self.prefix_length, 2, self.hidden_dim)
            prefix_key = prefix[0, :, 0, :]  # [prefix_length, hidden_dim]
            prefix_value = prefix[0, :, 1, :]  # [prefix_length, hidden_dim]
        else:
            prefix_key = prefix[0]  # [prefix_length, hidden_dim]
            prefix_value = prefix[1]  # [prefix_length, hidden_dim]

        # Reshape for multi-head attention
        # [prefix_length, num_heads, head_dim]
        prefix_key = prefix_key.view(self.prefix_length, self.num_heads, self.head_dim)
        prefix_value = prefix_value.view(self.prefix_length, self.num_heads, self.head_dim)

        # Transpose to [num_heads, prefix_length, head_dim]
        prefix_key = prefix_key.transpose(0, 1)
        prefix_value = prefix_value.transpose(0, 1)

        # Apply dropout
        prefix_key = self.dropout(prefix_key)
        prefix_value = self.dropout(prefix_value)

        return prefix_key, prefix_value


class AdapterConfig:
    """Configuration for adapter modules."""

    def __init__(
        self,
        adapter_type: str = "bottleneck",
        bottleneck_dim: int = 64,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        prefix_length: int = 20,
        dropout: float = 0.1,
        apply_to: str = "all",  # "all", "attention", "mlp"
    ):
        self.adapter_type = adapter_type
        self.bottleneck_dim = bottleneck_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.prefix_length = prefix_length
        self.dropout = dropout
        self.apply_to = apply_to


def add_adapters_to_model(
    model: nn.Module,
    adapter_config: AdapterConfig,
    freeze_base_model: bool = True,
) -> nn.Module:
    """
    Add adapter modules to a pre-trained model.

    Args:
        model: Pre-trained model
        adapter_config: Adapter configuration
        freeze_base_model: Whether to freeze base model parameters

    Returns:
        Model with adapters
    """
    if freeze_base_model:
        # Freeze all base model parameters
        for param in model.parameters():
            param.requires_grad = False
        logger.info("Base model parameters frozen")

    # Add adapters based on type
    if adapter_config.adapter_type == "bottleneck":
        _add_bottleneck_adapters(model, adapter_config)
    elif adapter_config.adapter_type == "lora":
        _add_lora_adapters(model, adapter_config)
    elif adapter_config.adapter_type == "prefix":
        _add_prefix_tuning(model, adapter_config)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_config.adapter_type}")

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    return model


def _add_bottleneck_adapters(model: nn.Module, config: AdapterConfig):
    """Add bottleneck adapters to transformer layers."""
    # This is a placeholder - actual implementation depends on model architecture
    # You would iterate through transformer layers and add adapters after MLP/attention
    logger.info(f"Adding bottleneck adapters with dim={config.bottleneck_dim}")
    # TODO: Implement for specific model architecture


def _add_lora_adapters(model: nn.Module, config: AdapterConfig):
    """Replace linear layers with LoRA layers."""
    logger.info(f"Adding LoRA adapters with rank={config.lora_rank}")
    # TODO: Implement for specific model architecture


def _add_prefix_tuning(model: nn.Module, config: AdapterConfig):
    """Add prefix tuning to the model."""
    logger.info(f"Adding prefix tuning with length={config.prefix_length}")
    # TODO: Implement for specific model architecture
