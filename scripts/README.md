# SE+ST Upgrade Training Scripts

Complete training pipeline for SE+ST model optimization.

## 📋 Quick Start

All scripts are located in the `scripts/` directory.

### Make scripts executable
```bash
chmod +x scripts/*.sh
```

---

## 🚀 Training Pipeline

### **Option 1: Quick Optimization (Recommended - 1-2 days)**

```bash
# 1. Train baseline model (40K steps, ~8 hours)
./scripts/train_baseline.sh

# 2. Test RL optimization (~10 hours)
./scripts/train_rl.sh

# 3. (Optional) Test AdapterTune (~5 hours)
./scripts/train_adaptertune.sh
```

### **Option 2: Full Optimization (3-4 days)**

```bash
# 1. Train baseline (40K steps)
./scripts/train_baseline.sh

# 2. AdapterTune fine-tuning
./scripts/train_adaptertune.sh

# 3. RL optimization on adapter model
./scripts/train_rl_from_adapter.sh

# 4. (Optional) AutoTune for best hyperparameters
./scripts/train_autotune.sh
```

### **Option 3: From Scratch with AutoTune (1 week)**

```bash
# 1. AutoTune to find best hyperparameters (~2-3 days)
./scripts/train_autotune.sh

# 2. Train with best config (80K steps, ~16 hours)
python -m cli.train \
  --config-path competition/autotune_v1 \
  --config-name best_config \
  training.max_steps=80000

# 3. AdapterTune on best model
./scripts/train_adaptertune.sh

# 4. Final RL optimization
./scripts/train_rl_from_adapter.sh
```

---

## 📝 Script Descriptions

### `train_baseline.sh`
- **Purpose**: Train base SE+ST model
- **Time**: ~8 hours (40K steps)
- **Output**: `competition/baseline_40k/final_model.ckpt`
- **GPU**: 1× A100

### `train_rl.sh`
- **Purpose**: RL fine-tuning with EXACT competition scoring
- **Time**: ~10 hours (100 epochs)
- **Output**: `competition/rl_optimized/best_model.ckpt`
- **GPU**: 1× A100
- **Metrics**: Directly optimizes DES + PDS + MAE using policy gradient

### `train_adaptertune.sh`
- **Purpose**: Parameter-efficient fine-tuning with LoRA
- **Time**: ~5 hours (20K steps)
- **Output**: `competition/adaptertune/final_adapters.pt`
- **GPU**: 1× A100
- **Benefit**: 95% parameter reduction, 3-5× faster

### `train_rl_from_adapter.sh`
- **Purpose**: RL fine-tuning on adapter-tuned model
- **Time**: ~10 hours (100 epochs)
- **Output**: `competition/rl_from_adapter/best_model.ckpt`
- **GPU**: 1× A100

### `train_autotune.sh`
- **Purpose**: Automated hyperparameter search
- **Time**: ~2-3 days (50 trials)
- **Output**: `competition/autotune_v1/best_config.yaml`
- **GPU**: 1× A100

---

## ⚙️ Configuration

All scripts use these default paths (edit scripts if yours differ):

```bash
# Data paths
TOML_CONFIG="/data/starter.toml"
ESM2_FEATURES="/data/ESM2_pert_features.pt"

# SE model paths
SE_MODEL_PATH="SE-600M"
SE_CHECKPOINT_PATH="SE-600M/se600m_epoch15.ckpt"

# Working directory
cd /workspace/se_st_upgrade
```

### To customize paths:
1. Edit the script you want to run
2. Update the variables at the top of the script
3. Save and run

---

## 📊 Expected Performance

| Method | Improvement | Training Time | Parameters Trained |
|--------|------------|---------------|-------------------|
| Baseline | - | 8h (40K) | 100% |
| + RL | +3-10% | +10h | 0% (strategy only) |
| + AdapterTune | +5-15% | +5h | ~5% |
| + AutoTune | +10-20% | +60h | 100% (new model) |

---

## 🔧 Troubleshooting

### Script fails with "command not found"
```bash
# Make sure you're in se_st_upgrade directory
cd /workspace/se_st_upgrade

# Make scripts executable
chmod +x scripts/*.sh
```

### "Checkpoint not found" error
```bash
# Check if previous step completed
ls competition/baseline_40k/final_model.ckpt

# Or update MODEL_CHECKPOINT in the script
```

### CUDA out of memory
```bash
# Reduce batch size in the script
# For baseline: BATCH_SIZE=4
# For adapter: BATCH_SIZE=8
# For RL: BATCH_SIZE=64
```

### Slow training
```bash
# Check GPU utilization
nvidia-smi

# Reduce num_workers if CPU is bottleneck
NUM_WORKERS=2
```

---

## 📈 Monitoring Training

### View logs
```bash
# TensorBoard (if enabled)
tensorboard --logdir competition/

# Or check output directory
ls -lh competition/baseline_40k/
```

### Check progress
```bash
# Watch training progress
tail -f competition/baseline_40k/train.log

# Check GPU usage
watch -n 1 nvidia-smi
```

---

## 💾 Output Files

Each script saves outputs to `competition/<experiment_name>/`:

```
competition/
├── baseline_40k/
│   ├── final_model.ckpt          # Trained model
│   ├── checkpoints/               # Intermediate checkpoints
│   └── config.yaml                # Training config
├── rl_optimized/
│   ├── best_model.ckpt            # Best RL-optimized model
│   ├── final_model.ckpt           # Final RL-optimized model
│   └── training_stats.pt          # Training statistics
├── adaptertune/
│   ├── final_adapters.pt          # Adapter weights (small!)
│   └── final_model.ckpt           # Full model (optional)
└── autotune_v1/
    ├── best_config.yaml           # Best hyperparameters
    ├── best_params.json           # Best parameters
    └── optimization_history.html  # Visualization
```

---

## 🎯 Recommended Workflow

For best results, follow this sequence:

```bash
# Week 1: Quick iteration
Day 1: ./scripts/train_baseline.sh
Day 2: ./scripts/train_rl.sh
Day 3: Evaluate and decide next steps

# Week 2: If needed, deeper optimization
Day 4-5: ./scripts/train_adaptertune.sh
Day 6: ./scripts/train_rl_from_adapter.sh
Day 7: Final evaluation and submission

# Optional: If pursuing maximum performance
Week 3-4: ./scripts/train_autotune.sh
Then: Retrain with best config
```

---

## 📞 Support

If you encounter issues:

1. Check this README
2. Review the script comments
3. Check `OPTIMIZATION_GUIDE.md` for detailed explanations
4. Verify your data paths are correct

---

**Good luck with your training!** 🚀
