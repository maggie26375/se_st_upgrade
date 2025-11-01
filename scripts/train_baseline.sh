#!/bin/bash
# SE+ST Upgrade - Baseline Training Script
# Train base model with 80K steps (baseline for all comparisons)

set -e  # Exit on error

echo "========================================="
echo "SE+ST Upgrade - Baseline Training"
echo "========================================="
echo ""

# Change to se_st_upgrade directory
cd /workspace/se_st_upgrade || {
    echo "Error: /workspace/se_st_upgrade not found"
    echo "Please adjust the path to your se_st_upgrade directory"
    exit 1
}

echo "Working directory: $(pwd)"
echo "Starting training at: $(date)"
echo ""

# Set PYTHONPATH so Python can find se_st_upgrade package
export PYTHONPATH=/workspace/se_st_upgrade:$PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"
echo ""

# Training parameters
MAX_STEPS=80000  # 80K for best results (your previous successful training)
BATCH_SIZE=8
VAL_CHECK_INTERVAL=2000  # Check validation every 2K steps
NUM_WORKERS=4

# Data paths
TOML_CONFIG="/data/starter.toml"
ESM2_FEATURES="/data/ESM2_pert_features.pt"

# Model parameters
INPUT_DIM=18080
HIDDEN_DIM=512
OUTPUT_DIM=18080
PERT_DIM=5120
SE_MODEL_PATH="SE-600M"
SE_CHECKPOINT_PATH="SE-600M/se600m_epoch15.ckpt"

# Output configuration
OUTPUT_DIR="competition"
EXPERIMENT_NAME="baseline_80k"  # Changed from baseline_40k

echo "Configuration:"
echo "  Max steps: $MAX_STEPS"
echo "  Batch size: $BATCH_SIZE"
echo "  Data config: $TOML_CONFIG"
echo "  ESM2 features: $ESM2_FEATURES"
echo "  Output: $OUTPUT_DIR/$EXPERIMENT_NAME"
echo ""

# Run training
python -m cli.train \
  data.kwargs.toml_config_path="$TOML_CONFIG" \
  data.kwargs.perturbation_features_file="$ESM2_FEATURES" \
  data.kwargs.num_workers=$NUM_WORKERS \
  training.max_steps=$MAX_STEPS \
  training.batch_size=$BATCH_SIZE \
  training.val_check_interval=$VAL_CHECK_INTERVAL \
  model.kwargs.input_dim=$INPUT_DIM \
  model.kwargs.hidden_dim=$HIDDEN_DIM \
  model.kwargs.output_dim=$OUTPUT_DIM \
  model.kwargs.pert_dim=$PERT_DIM \
  model.kwargs.se_model_path="$SE_MODEL_PATH" \
  model.kwargs.se_checkpoint_path="$SE_CHECKPOINT_PATH" \
  output_dir="$OUTPUT_DIR" \
  name="$EXPERIMENT_NAME"

echo ""
echo "========================================="
echo "✅ Training completed!"
echo "========================================="
echo "Checkpoint saved to: $OUTPUT_DIR/$EXPERIMENT_NAME/final_model.ckpt"
echo "Training finished at: $(date)"
echo ""
echo "Next steps:"
echo "  1. Test with RL: ./scripts/train_rl.sh"
echo "  2. Test with AdapterTune: ./scripts/train_adaptertune.sh"
echo ""
