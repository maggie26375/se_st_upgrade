#!/bin/bash
# SE+ST Upgrade - Continue from 160K to 200K
# Resume training from 160K checkpoint with full state restoration

set -e  # Exit on error

echo "========================================="
echo "SE+ST Upgrade - Continue to 200K"
echo "========================================="
echo ""

# Change to se_st_upgrade directory
cd /workspace/se_st_upgrade || {
    echo "Error: /workspace/se_st_upgrade not found"
    echo "Please adjust the path to your se_st_upgrade directory"
    exit 1
}

echo "Working directory: $(pwd)"
echo "Continuing training at: $(date)"
echo ""

# Set PYTHONPATH
export PYTHONPATH=/workspace/se_st_upgrade:$PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"
echo ""

# Training parameters
MAX_STEPS=200000  # Continue to 200K (additional 40K steps)
BATCH_SIZE=8
VAL_CHECK_INTERVAL=2000
NUM_WORKERS=4

# Checkpoint to resume from
CHECKPOINT="competition/baseline_160k/final_model.ckpt"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found at $CHECKPOINT"
    echo ""
    echo "Please ensure baseline_160k training completed successfully"
    exit 1
fi

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
EXPERIMENT_NAME="baseline_200k"

echo "Configuration:"
echo "  ⚡ RESUMING from: $CHECKPOINT"
echo "  🎯 Target steps: $MAX_STEPS (current: 160K, additional: 40K)"
echo "  📊 Batch size: $BATCH_SIZE"
echo "  💾 Output: $OUTPUT_DIR/$EXPERIMENT_NAME"
echo ""
echo "Training Progress Summary:"
echo "  ✅  80K: val_loss = 35.32"
echo "  ✅ 120K: val_loss = 33.67 (↓ 1.65)"
echo "  ✅ 160K: val_loss = 32.34 (↓ 1.33)"
echo "  🎯 200K: val_loss ≈ 31.0-31.3 (estimated ↓ 1.0-1.3)"
echo ""
echo "This will restore:"
echo "  ✅ Model weights"
echo "  ✅ Optimizer state"
echo "  ✅ Learning rate scheduler"
echo "  ✅ Global step counter (will start from ~160,000)"
echo "  ✅ Epoch counter"
echo ""

# Run training with checkpoint resumption
python run_train.py \
  model.checkpoint="$CHECKPOINT" \
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
echo "Final model saved to: $OUTPUT_DIR/$EXPERIMENT_NAME/final_model.ckpt"
echo "Training finished at: $(date)"
echo ""
echo "📈 Training Evolution:"
echo "   80K →  120K: -1.65 loss"
echo "  120K →  160K: -1.33 loss"
echo "  160K →  200K: -1.0~1.3 loss (estimated)"
echo "  Total improvement from 80K: ~4.0-4.3 points (11-12%)"
echo ""
echo "Next steps:"
echo "  1. Run inference to get competition score"
echo "  2. Test RL optimization: ./scripts/train_rl.sh"
echo "  3. Test AdapterTune: ./scripts/train_adaptertune.sh"
echo ""
