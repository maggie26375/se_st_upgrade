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

### **Option 1: Systematic Testing (Recommended - find what works best)**

This approach trains a strong 80K baseline, then tests each optimization method separately.

```bash
# 1. Train baseline model (80K steps, ~16 hours)
./scripts/train_baseline.sh
# → competition/baseline_80k/final_model.ckpt
# → Run inference → Record baseline score

# 2. Test RL optimization (~10 hours)
./scripts/train_rl.sh  # Starts from baseline_80k
# → competition/rl_optimized/best_model.ckpt
# → Run inference → Compare with baseline (RL improvement?)

# 3. Test AdapterTune (~5 hours)
./scripts/train_adaptertune.sh  # Starts from baseline_80k
# → competition/adaptertune/final_model.ckpt
# → Run inference → Compare with baseline (Adapter improvement?)

# 4. Test AdapterTune + RL (~10 hours)
./scripts/train_rl_from_adapter.sh  # Starts from adaptertune
# → competition/rl_from_adapter/best_model.ckpt
# → Run inference → Best combination?
```

**Why this approach?**
- ✅ Know which method actually helps
- ✅ Compare each method fairly (same 80K baseline)
- ✅ Can stop early if one method doesn't help
- ✅ Saves time by not running useless optimizations

### **Option 2: Hyperparameter Search (1 week - for maximum performance)**

Only do this if Option 1 results are good but you want to squeeze out more performance.

```bash
# 1. AutoTune to find best hyperparameters (~2-3 days)
./scripts/train_autotune.sh
# → competition/autotune_v1/best_config.yaml

# 2. Train with best config (80K steps, ~16 hours)
python -m cli.train \
  --config-path competition/autotune_v1 \
  --config-name best_config \
  training.max_steps=80000
# → competition/autotune_80k/final_model.ckpt
# → Run inference → Compare with baseline_80k

# 3. If AutoTune model is better, apply optimizations on it
#    (AdapterTune, RL, etc.)
```

---

## 📝 Script Descriptions

### `train_baseline.sh`
- **Purpose**: Train base SE+ST model (baseline for all comparisons)
- **Time**: ~16 hours (80K steps)
- **Output**: `competition/baseline_80k/final_model.ckpt`
- **GPU**: 1× A100
- **Note**: This is your baseline - run inference and record the score!

### `train_rl.sh`
- **Purpose**: RL fine-tuning with EXACT competition scoring
- **Time**: ~10 hours (100 epochs)
- **Input**: `baseline_80k/final_model.ckpt`
- **Output**: `competition/rl_optimized/best_model.ckpt`
- **GPU**: 1× A100
- **Metrics**: Directly optimizes DES + PDS + MAE using policy gradient
- **Note**: Run inference and compare with baseline to see if RL helps!

### `train_adaptertune.sh`
- **Purpose**: Parameter-efficient fine-tuning with LoRA
- **Time**: ~5 hours (20K steps)
- **Input**: `baseline_80k/final_model.ckpt`
- **Output**: `competition/adaptertune/final_model.ckpt`
- **GPU**: 1× A100
- **Benefit**: 95% parameter reduction, 3-5× faster
- **Note**: Run inference and compare with baseline!

### `train_rl_from_adapter.sh`
- **Purpose**: RL fine-tuning on adapter-tuned model
- **Time**: ~10 hours (100 epochs)
- **Input**: `adaptertune/final_model.ckpt`
- **Output**: `competition/rl_from_adapter/best_model.ckpt`
- **GPU**: 1× A100
- **Note**: Tests if AdapterTune + RL combination is best

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
├── baseline_80k/                  # ⭐ YOUR BASELINE
│   ├── final_model.ckpt          # Trained model (80K steps)
│   ├── checkpoints/               # Intermediate checkpoints
│   └── config.yaml                # Training config
├── rl_optimized/                  # Test 1: RL on baseline
│   ├── best_model.ckpt            # Best RL-optimized model
│   ├── final_model.ckpt           # Final RL-optimized model
│   └── training_stats.pt          # Training statistics
├── adaptertune/                   # Test 2: Adapter on baseline
│   ├── final_adapters.pt          # Adapter weights (small!)
│   └── final_model.ckpt           # Full model
├── rl_from_adapter/               # Test 3: RL on adapter
│   ├── best_model.ckpt            # Best combined model
│   └── final_model.ckpt           # Final combined model
└── autotune_v1/                   # Optional: Hyperparameter search
    ├── best_config.yaml           # Best hyperparameters
    ├── best_params.json           # Best parameters
    └── optimization_history.html  # Visualization
```

---

## 🎯 Recommended Workflow

### **Step-by-Step Testing Approach** (Most Scientific)

```bash
# Day 1-2: Train baseline (80K steps, ~16 hours)
./scripts/train_baseline.sh

# Day 2: Inference baseline
se-st-infer \
  --checkpoint competition/baseline_80k/final_model.ckpt \
  --adata /data/competition_val_template.h5ad \
  --output baseline_pred.h5ad \
  --perturbation-features /data/ESM2_pert_features.pt \
  --se-model-path SE-600M \
  --batch-size 16 \
  --device cuda

# Upload to competition → Record score (e.g., 85 points)
# This is your baseline!

# Day 3: Test RL (~10 hours)
./scripts/train_rl.sh
# → Inference → Compare score (e.g., 87 points → +2!)
# Decision: RL helps? Continue using it.

# Day 4: Test AdapterTune (~5 hours)
./scripts/train_adaptertune.sh  # Starts from baseline_80k, NOT RL!
# → Inference → Compare with baseline (e.g., 88 points → +3!)
# Decision: Adapter better than RL alone!

# Day 5: Test Adapter + RL (~10 hours)
./scripts/train_rl_from_adapter.sh
# → Inference → Compare (e.g., 90 points → +5!)
# Decision: Best combination found!

# Optional: AutoTune if you want to squeeze more
Week 2-3: ./scripts/train_autotune.sh
# Only do this if current results are good but not winning
```

### **Why This Workflow?**

1. **Baseline first** (80K steps) - Your reference point
2. **Test each method separately** - Know what works
3. **Test combinations** - Find best stack
4. **Run inference after each step** - Immediate feedback
5. **Compare scores** - Data-driven decisions

### **Key Principle: Compare Everything to Baseline**

| Method | Checkpoint | Score | Improvement |
|--------|-----------|-------|-------------|
| Baseline (80K) | `baseline_80k/final_model.ckpt` | 85 | - |
| + RL | `rl_optimized/best_model.ckpt` | 87 | +2 |
| + Adapter | `adaptertune/final_model.ckpt` | 88 | +3 |
| + Adapter + RL | `rl_from_adapter/best_model.ckpt` | 90 | +5 ✅ |

Now you know: **AdapterTune + RL = Best!**

---

## 📞 Support

If you encounter issues:

1. Check this README
2. Review the script comments
3. Check `OPTIMIZATION_GUIDE.md` for detailed explanations
4. Verify your data paths are correct

---

**Good luck with your training!** 🚀
