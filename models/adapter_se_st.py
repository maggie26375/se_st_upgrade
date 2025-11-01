"""
SE+ST Combined Model with Adapter-based fine-tuning
"""

import logging
from typing import Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn

from .se_st_combined import SE_ST_CombinedModel
from .adapters import BottleneckAdapter, LoRALinear, AdapterConfig

logger = logging.getLogger(__name__)


class AdapterSE_ST_Model(SE_ST_CombinedModel):
    """
    SE+ST model with parameter-efficient adapter-based fine-tuning.

    Supports:
    1. Bottleneck adapters in transformer layers
    2. LoRA for attention/MLP layers
    3. Selective unfreezing of specific layers
    """

    def __init__(
        self,
        # Base model parameters
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        se_model_path: str,
        se_checkpoint_path: str,
        # Adapter parameters
        adapter_type: str = "bottleneck",  # bottleneck, lora, hybrid
        bottleneck_dim: int = 64,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        adapter_dropout: float = 0.1,
        freeze_st_model: bool = True,
        unfreeze_layers: Optional[list] = None,  # Layers to unfreeze
        # Other parameters
        **kwargs,
    ):
        """
        Initialize Adapter SE+ST model.

        Args:
            adapter_type: Type of adapter (bottleneck, lora, hybrid)
            bottleneck_dim: Bottleneck dimension for adapter
            lora_rank: Rank for LoRA
            lora_alpha: Alpha scaling for LoRA
            adapter_dropout: Dropout for adapters
            freeze_st_model: Whether to freeze ST model base parameters
            unfreeze_layers: List of layer indices to unfreeze
        """
        # Initialize base model
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            se_model_path=se_model_path,
            se_checkpoint_path=se_checkpoint_path,
            **kwargs,
        )

        self.adapter_type = adapter_type
        self.bottleneck_dim = bottleneck_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.adapter_dropout = adapter_dropout
        self.freeze_st_model = freeze_st_model
        self.unfreeze_layers = unfreeze_layers or []

        # Add adapters to ST model
        self._add_adapters_to_st_model()

        # Log adapter configuration
        self._log_adapter_info()

    def _add_adapters_to_st_model(self):
        """Add adapter modules to ST model."""
        if self.freeze_st_model:
            # Freeze ST model parameters
            for param in self.st_model.parameters():
                param.requires_grad = False
            logger.info("ST model parameters frozen")

        # Add adapters based on type
        if self.adapter_type == "bottleneck":
            self._add_bottleneck_adapters()
        elif self.adapter_type == "lora":
            self._add_lora_to_st_model()
        elif self.adapter_type == "hybrid":
            self._add_bottleneck_adapters()
            self._add_lora_to_st_model()
        else:
            raise ValueError(f"Unknown adapter type: {self.adapter_type}")

        # Selectively unfreeze layers if specified
        if self.unfreeze_layers:
            self._unfreeze_specific_layers()

    def _add_bottleneck_adapters(self):
        """Add bottleneck adapters to ST model transformer layers."""
        # Access ST model's transformer backbone
        if not hasattr(self.st_model, 'transformer_backbone'):
            logger.warning("ST model does not have transformer_backbone, skipping adapters")
            return

        transformer = self.st_model.transformer_backbone
        hidden_dim = self.st_hidden_dim

        # Add adapters after each transformer layer
        num_layers = len(transformer.layers) if hasattr(transformer, 'layers') else 0

        if num_layers == 0:
            logger.warning("Could not find transformer layers to add adapters")
            return

        # Create adapter modules
        self.adapters = nn.ModuleList()
        for layer_idx in range(num_layers):
            adapter = BottleneckAdapter(
                input_dim=hidden_dim,
                bottleneck_dim=self.bottleneck_dim,
                dropout=self.adapter_dropout,
            )
            self.adapters.append(adapter)

        logger.info(f"Added {num_layers} bottleneck adapters (dim={self.bottleneck_dim})")

        # Register forward hooks to insert adapters
        self._register_adapter_hooks()

    def _register_adapter_hooks(self):
        """Register forward hooks to insert adapter modules."""
        if not hasattr(self.st_model, 'transformer_backbone'):
            return

        transformer = self.st_model.transformer_backbone

        if not hasattr(transformer, 'layers'):
            return

        # Hook to insert adapter after each layer
        def create_adapter_hook(adapter_module):
            def hook(module, input, output):
                # Apply adapter to output
                if isinstance(output, tuple):
                    # If output is tuple (hidden_states, ...), adapt hidden_states
                    hidden_states = output[0]
                    adapted = adapter_module(hidden_states)
                    return (adapted,) + output[1:]
                else:
                    return adapter_module(output)
            return hook

        # Register hooks
        for layer_idx, layer in enumerate(transformer.layers):
            if layer_idx < len(self.adapters):
                layer.register_forward_hook(create_adapter_hook(self.adapters[layer_idx]))

        logger.info(f"Registered adapter hooks for {len(self.adapters)} layers")

    def _add_lora_to_st_model(self):
        """Add LoRA to ST model's linear layers."""
        # Replace attention Q, K, V projections with LoRA
        if not hasattr(self.st_model, 'transformer_backbone'):
            logger.warning("ST model does not have transformer_backbone, skipping LoRA")
            return

        transformer = self.st_model.transformer_backbone

        if not hasattr(transformer, 'layers'):
            return

        num_replaced = 0

        # Replace linear layers with LoRA layers
        for layer_idx, layer in enumerate(transformer.layers):
            # Replace attention layers
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn

                # Replace Q, K, V projections
                for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    if hasattr(attn, name):
                        linear = getattr(attn, name)
                        lora_linear = LoRALinear(
                            in_features=linear.in_features,
                            out_features=linear.out_features,
                            rank=self.lora_rank,
                            alpha=self.lora_alpha,
                            dropout=self.adapter_dropout,
                        )
                        # Copy weights from original linear
                        lora_linear.linear.weight.data = linear.weight.data.clone()
                        if linear.bias is not None:
                            lora_linear.linear.bias.data = linear.bias.data.clone()

                        # Replace
                        setattr(attn, name, lora_linear)
                        num_replaced += 1

        logger.info(f"Replaced {num_replaced} linear layers with LoRA (rank={self.lora_rank})")

    def _unfreeze_specific_layers(self):
        """Unfreeze specific transformer layers."""
        if not hasattr(self.st_model, 'transformer_backbone'):
            return

        transformer = self.st_model.transformer_backbone

        if not hasattr(transformer, 'layers'):
            return

        for layer_idx in self.unfreeze_layers:
            if layer_idx < len(transformer.layers):
                for param in transformer.layers[layer_idx].parameters():
                    param.requires_grad = True
                logger.info(f"Unfrozen layer {layer_idx}")

    def _log_adapter_info(self):
        """Log adapter configuration and parameter statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        adapter_params = 0

        # Count adapter parameters
        if hasattr(self, 'adapters'):
            adapter_params += sum(p.numel() for p in self.adapters.parameters())

        # Count LoRA parameters
        for module in self.modules():
            if isinstance(module, LoRALinear):
                adapter_params += module.lora_A.numel() + module.lora_B.numel()

        logger.info("=" * 60)
        logger.info("Adapter SE+ST Model Configuration")
        logger.info("=" * 60)
        logger.info(f"Adapter type: {self.adapter_type}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Adapter parameters: {adapter_params:,}")
        logger.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        logger.info(f"Parameter reduction: {100 * (1 - trainable_params / total_params):.2f}%")
        logger.info("=" * 60)

    def save_adapters(self, save_path: str):
        """
        Save only adapter parameters (not the full model).

        Args:
            save_path: Path to save adapter weights
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        adapter_state = {}

        # Save bottleneck adapters
        if hasattr(self, 'adapters'):
            adapter_state['adapters'] = self.adapters.state_dict()

        # Save LoRA parameters
        lora_state = {}
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                lora_state[name] = {
                    'lora_A': module.lora_A.data,
                    'lora_B': module.lora_B.data,
                }
        if lora_state:
            adapter_state['lora'] = lora_state

        # Save adapter config
        adapter_state['config'] = {
            'adapter_type': self.adapter_type,
            'bottleneck_dim': self.bottleneck_dim,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
        }

        torch.save(adapter_state, save_path)
        logger.info(f"Adapter weights saved to {save_path}")

    def load_adapters(self, load_path: str):
        """
        Load adapter parameters.

        Args:
            load_path: Path to load adapter weights from
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Adapter weights not found: {load_path}")

        adapter_state = torch.load(load_path, map_location='cpu')

        # Load bottleneck adapters
        if 'adapters' in adapter_state and hasattr(self, 'adapters'):
            self.adapters.load_state_dict(adapter_state['adapters'])
            logger.info("Bottleneck adapters loaded")

        # Load LoRA parameters
        if 'lora' in adapter_state:
            for name, module in self.named_modules():
                if isinstance(module, LoRALinear) and name in adapter_state['lora']:
                    module.lora_A.data = adapter_state['lora'][name]['lora_A']
                    module.lora_B.data = adapter_state['lora'][name]['lora_B']
            logger.info("LoRA parameters loaded")

        logger.info(f"Adapters loaded from {load_path}")


def load_model_with_adapters(
    base_checkpoint: str,
    adapter_checkpoint: str,
    **model_kwargs,
) -> AdapterSE_ST_Model:
    """
    Load SE+ST model with adapters from checkpoints.

    Args:
        base_checkpoint: Path to base model checkpoint
        adapter_checkpoint: Path to adapter weights
        **model_kwargs: Additional model arguments

    Returns:
        Model with adapters loaded
    """
    # Load base model
    model = AdapterSE_ST_Model(**model_kwargs)

    # Load base model weights if provided
    if Path(base_checkpoint).exists():
        checkpoint = torch.load(base_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info(f"Base model loaded from {base_checkpoint}")

    # Load adapter weights
    model.load_adapters(adapter_checkpoint)

    return model
