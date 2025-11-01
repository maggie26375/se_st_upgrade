#!/bin/bash
# SE+ST Upgrade - Continue from 120K to 160K
# Resume training from 120K checkpoint with full state restoration

set -e  # Exit on error

echo "========================================="
echo "SE+ST Upgrade - Continue to 160K"
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
MAX_STEPS=160000  # Continue to 160K (additional 40K steps)
BATCH_SIZE=8
VAL_CHECK_INTERVAL=2000
NUM_WORKERS=4

# Checkpoint to resume from
CHECKPOINT="competition/baseline_120k/final_model.ckpt"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found at $CHECKPOINT"
    echo ""
    echo "Please ensure baseline_120k training completed successfully"
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
EXPERIMENT_NAME="baseline_160k"

echo "Configuration:"
echo "  ⚡ RESUMING from: $CHECKPOINT"
echo "  🎯 Target steps: $MAX_STEPS (current: ~120K, additional: ~40K)"
echo "  📊 Batch size: $BATCH_SIZE"
echo "  💾 Output: $OUTPUT_DIR/$EXPERIMENT_NAME"
echo ""
echo "This will restore:"
echo "  ✅ Model weights"
echo "  ✅ Optimizer state"
echo "  ✅ Learning rate scheduler"
echo "  ✅ Global step counter (will start from ~120,000)"
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
echo "📈 Expected improvement:"
echo "  120K baseline: val_loss ≈ 33.67"
echo "  160K target:   val_loss ≈ 32.0-32.5 (estimated)"
echo ""
