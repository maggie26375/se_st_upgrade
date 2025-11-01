# SE+ST Model Optimization Guide

Complete guide for the three-phase optimization strategy: **AutoTune → AdapterTune → RL**

---

## 📋 Overview

This optimization pipeline is designed to progressively improve the SE+ST perturbation prediction model:

1. **Phase 1: AutoTune** - Automated hyperparameter search to find optimal model configuration
2. **Phase 2: AdapterTune** - Parameter-efficient fine-tuning for specific tasks/cell types
3. **Phase 3: RL** - Reinforcement learning to optimize prediction strategies

---

## 🔍 Phase 1: AutoTune (Hyperparameter Optimization)

### What it does
- Automatically searches for the best hyperparameters using Bayesian optimization (Optuna)
- Explores: model architecture, learning rates, batch sizes, loss functions, etc.
- Runs multiple trials with early stopping for efficiency

### How to use

```bash
# Basic usage
python -m se_st_upgrade.cli.autotune

# Custom configuration
python -m se_st_upgrade.cli.autotune \
  autotune.n_trials=100 \
  autotune.resources.max_steps=10000 \
  data.kwargs.toml_config_path=/path/to/data.toml \
  data.kwargs.perturbation_features_file=/path/to/features.pt
```

### Configuration
Edit `configs/autotune.yaml` to customize:

- **Search space**: Which hyperparameters to optimize
- **Number of trials**: How many configurations to test
- **Resources**: Training budget per trial
- **Pruning**: Early stopping for unpromising trials

### Output
```
outputs/se_st_autotune/
├── best_config.yaml           # Best hyperparameter configuration
├── best_params.json           # Best parameters (JSON)
├── study_summary.json         # Optimization summary
├── all_trials.json            # All trial results
├── optimization_history.html  # Visualization
└── param_importances.html     # Parameter importance plot
```

### Next steps
```bash
# Train with best hyperparameters
python -m se_st_upgrade.cli.train \
  --config-path outputs/se_st_autotune \
  --config-name best_config
```

---

## 🔧 Phase 2: AdapterTune (Parameter-Efficient Fine-Tuning)

### What it does
- Adds small adapter modules to the model (bottleneck adapters, LoRA, or both)
- Freezes the base model and only trains adapters (~5% of parameters)
- Enables fast fine-tuning for specific cell types or conditions
- 3-5x faster training, 95%+ memory savings

### Adapter types

1. **Bottleneck Adapters**: Small MLP modules inserted after transformer layers
2. **LoRA (Low-Rank Adaptation)**: Low-rank decomposition of attention weights
3. **Hybrid**: Combines both approaches

### How to use

```bash
# Basic adapter fine-tuning
python -m se_st_upgrade.cli.adaptertune \
  data.kwargs.toml_config_path=/path/to/data.toml \
  data.kwargs.perturbation_features_file=/path/to/features.pt \
  model.checkpoint=/path/to/pretrained_model.ckpt

# Use LoRA adapters
python -m se_st_upgrade.cli.adaptertune \
  model.kwargs.adapter_type=lora \
  model.kwargs.lora_rank=16 \
  model.kwargs.lora_alpha=16.0

# Hybrid adapters
python -m se_st_upgrade.cli.adaptertune \
  model.kwargs.adapter_type=hybrid \
  model.kwargs.bottleneck_dim=64 \
  model.kwargs.lora_rank=16

# Unfreeze specific layers
python -m se_st_upgrade.cli.adaptertune \
  model.kwargs.unfreeze_layers=[10,11]
```

### Configuration
Edit `configs/adaptertune.yaml`:

```yaml
model:
  kwargs:
    adapter_type: bottleneck  # bottleneck, lora, hybrid
    bottleneck_dim: 64        # Adapter bottleneck dimension
    lora_rank: 16             # LoRA rank
    lora_alpha: 16.0          # LoRA scaling
    adapter_dropout: 0.1      # Dropout rate
    freeze_st_model: true     # Freeze base model
    unfreeze_layers: []       # Optionally unfreeze layers
```

### Output
```
outputs/se_st_adaptertune/
├── adapters/
│   └── step=X-val_loss=Y_adapters.pt  # Adapter weights only
├── final_adapters.pt          # Final adapter weights
├── config.yaml                # Training configuration
└── checkpoints/               # Full checkpoints (if enabled)
```

### Loading adapters

```python
from se_st_upgrade.models.adapter_se_st import load_model_with_adapters

model = load_model_with_adapters(
    base_checkpoint="path/to/base_model.ckpt",
    adapter_checkpoint="path/to/adapters.pt",
    **model_kwargs
)
```

---

## 🎮 Phase 3: RL (Reinforcement Learning Optimization)

### What it does
- Uses RL (PPO algorithm) to learn optimal prediction strategies
- Learns to select the best control cells for comparison
- Optimizes confidence weighting for predictions
- Improves beyond supervised learning by directly optimizing prediction quality

### How it works

**Environment:**
- **State**: Perturbation embedding + current control cell state + metadata
- **Action**: [select_control_idx, confidence_weight]
- **Reward**: Negative prediction error (MSE, energy distance, or correlation)

**Agent:**
- PPO (Proximal Policy Optimization) with actor-critic architecture
- Learns both policy (what to do) and value function (how good is the state)

### How to use

```bash
# Basic RL training
python -m se_st_upgrade.cli.rltune \
  model.checkpoint=/path/to/pretrained_model.ckpt \
  data.kwargs.toml_config_path=/path/to/data.toml \
  data.kwargs.perturbation_features_file=/path/to/features.pt

# Custom RL settings
python -m se_st_upgrade.cli.rltune \
  rl.training.n_episodes=2000 \
  rl.agent.lr_actor=1e-4 \
  rl.agent.lr_critic=5e-4 \
  rl.env.reward_type=negative_energy

# Different reward types
python -m se_st_upgrade.cli.rltune \
  rl.env.reward_type=correlation  # correlation, negative_mse, negative_energy
```

### Configuration
Edit `configs/rltune.yaml`:

```yaml
rl:
  env:
    num_control_cells: 128
    max_episode_steps: 50
    reward_type: negative_mse

  agent:
    algorithm: ppo
    lr_actor: 3e-4
    lr_critic: 1e-3
    gamma: 0.99
    clip_ratio: 0.2

  training:
    n_episodes: 1000
    rollout_buffer_size: 2048
    n_epochs_per_update: 10
```

### Output
```
outputs/se_st_rltune/
├── best_agent.pt          # Best RL agent (by reward)
├── final_agent.pt         # Final RL agent
├── training_stats.pt      # Training statistics
├── agent_episode_X.pt     # Periodic checkpoints
└── config.yaml            # Configuration
```

### Using the RL agent

```python
from se_st_upgrade.models.rl_agent import PPOAgent
from se_st_upgrade.utils.rl_environment import create_rl_env

# Load agent
agent = PPOAgent(obs_dim=..., action_dim=...)
agent.load("outputs/se_st_rltune/best_agent.pt")

# Use for inference
env = create_rl_env(model, dataset)
obs, _ = env.reset()
action, _, _ = agent.select_action(obs, deterministic=True)
```

---

## 🚀 Complete Pipeline Example

### Step 1: Find best hyperparameters
```bash
python -m se_st_upgrade.cli.autotune \
  autotune.n_trials=50 \
  data.kwargs.toml_config_path=/data/train_config.toml \
  data.kwargs.perturbation_features_file=/data/ESM2_features.pt
```

### Step 2: Train base model with best config
```bash
python -m se_st_upgrade.cli.train \
  --config-path outputs/se_st_autotune \
  --config-name best_config \
  training.max_steps=50000
```

### Step 3: Fine-tune with adapters for specific cell type
```bash
python -m se_st_upgrade.cli.adaptertune \
  model.checkpoint=outputs/se_st_autotune/best_model.ckpt \
  model.kwargs.adapter_type=lora \
  data.kwargs.toml_config_path=/data/celltype_specific.toml \
  training.max_steps=10000
```

### Step 4: RL optimization
```bash
python -m se_st_upgrade.cli.rltune \
  model.checkpoint=outputs/se_st_adaptertune/final_model.ckpt \
  data.kwargs.toml_config_path=/data/train_config.toml \
  rl.training.n_episodes=1000
```

---

## 📊 Expected Performance Improvements

| Phase | Metric | Expected Improvement |
|-------|--------|---------------------|
| AutoTune | Val Loss | 10-20% reduction |
| AdapterTune | Training Speed | 3-5x faster |
| AdapterTune | Memory Usage | 95% reduction |
| RL | Prediction Quality | 5-15% improvement |

---

## 🛠️ Installation Requirements

```bash
# Core dependencies (already in requirements.txt)
pip install torch lightning hydra-core

# AutoTune dependencies
pip install optuna plotly

# AdapterTune (no additional dependencies)

# RL dependencies
pip install gymnasium tqdm
```

---

## 💡 Tips and Best Practices

### AutoTune
- Start with fewer trials (20-30) for quick exploration
- Use `autotune.resources.max_steps=5000` for faster trials
- Check `param_importances.html` to understand which parameters matter most

### AdapterTune
- Use bottleneck adapters for general fine-tuning
- Use LoRA for attention-heavy tasks
- Use hybrid for maximum flexibility
- Start with `bottleneck_dim=64` and `lora_rank=16`

### RL
- Pretrain the base model well before RL
- Start with simple reward (negative_mse)
- Monitor episode rewards - should increase over time
- Use `eval_frequency=50` to track progress
- RL can be unstable - save checkpoints frequently

---

## 🐛 Troubleshooting

### AutoTune
**Issue**: Trials failing with CUDA OOM
- Solution: Reduce `autotune.resources.max_steps` or exclude large `batch_size` from search space

**Issue**: All trials have similar performance
- Solution: Expand search space or check if model/data is the bottleneck

### AdapterTune
**Issue**: Adapters not improving performance
- Solution: Check that base model is frozen (`freeze_st_model=true`)
- Solution: Try different adapter types or increase adapter capacity

**Issue**: Training is still slow
- Solution: Verify adapters are actually frozen (check parameter count in logs)

### RL
**Issue**: Rewards not improving
- Solution: Check reward function is appropriate for task
- Solution: Reduce learning rates (`lr_actor`, `lr_critic`)
- Solution: Ensure base model is producing reasonable predictions

**Issue**: Training is unstable
- Solution: Reduce `clip_ratio` (try 0.1 instead of 0.2)
- Solution: Increase `batch_size` or `rollout_buffer_size`

---

## 📚 References

- **AutoTune**: [Optuna Documentation](https://optuna.org/)
- **Adapters**: [Parameter-Efficient Transfer Learning (Houlsby et al., 2019)](https://arxiv.org/abs/1902.00751)
- **LoRA**: [Low-Rank Adaptation (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- **PPO**: [Proximal Policy Optimization (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)

---

## 🎯 Summary

The three-phase optimization pipeline provides a systematic approach to improving your SE+ST model:

1. **AutoTune** finds the best architecture and hyperparameters
2. **AdapterTune** enables fast, efficient fine-tuning for specific tasks
3. **RL** optimizes prediction strategies beyond supervised learning

Each phase builds on the previous one, creating a powerful optimization workflow!
