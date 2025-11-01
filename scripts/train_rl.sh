#!/bin/bash
# SE+ST Upgrade - RL Optimization Script
# Fine-tune model using Policy Gradient with EXACT competition scoring

set -e  # Exit on error

echo "========================================="
echo "SE+ST Upgrade - RL Optimization"
echo "========================================="
echo ""

# Change to se_st_upgrade directory
cd /workspace/se_st_upgrade || {
    echo "Error: /workspace/se_st_upgrade not found"
    echo "Please adjust the path to your se_st_upgrade directory"
    exit 1
}

echo "Working directory: $(pwd)"
echo "Starting RL training at: $(date)"
echo ""

# Model checkpoint to optimize (update this path!)
MODEL_CHECKPOINT="competition/baseline_80k/final_model.ckpt"

# Check if checkpoint exists
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "❌ Error: Model checkpoint not found at $MODEL_CHECKPOINT"
    echo ""
    echo "Please either:"
    echo "  1. Run ./scripts/train_baseline.sh first, or"
    echo "  2. Update MODEL_CHECKPOINT variable in this script"
    exit 1
fi

# RL training parameters
N_EPISODES=100  # Training epochs
LEARNING_RATE=1e-5  # Low LR for fine-tuning
BATCH_SIZE=4
REWARD_TYPE="competition"  # Use EXACT competition scoring!

# Data paths
TOML_CONFIG="/data/starter.toml"
ESM2_FEATURES="/data/ESM2_pert_features.pt"
NUM_WORKERS=4

# Model parameters (must match base model)
INPUT_DIM=18080
HIDDEN_DIM=512
OUTPUT_DIM=18080
PERT_DIM=5120
SE_MODEL_PATH="SE-600M"
SE_CHECKPOINT_PATH="SE-600M/se600m_epoch15.ckpt"

# Output configuration
OUTPUT_DIR="competition"
EXPERIMENT_NAME="rl_optimized"

echo "Configuration:"
echo "  Base model: $MODEL_CHECKPOINT"
echo "  Episodes (epochs): $N_EPISODES"
echo "  Learning rate: $LEARNING_RATE"
echo "  Reward type: $REWARD_TYPE (EXACT competition scoring)"
echo "  Output: $OUTPUT_DIR/$EXPERIMENT_NAME"
echo ""
echo "⚠️  This will take approximately $(($N_EPISODES / 10)) hours on A100"
echo ""

# Run RL training
python -m cli.rltune \
  model.checkpoint="$MODEL_CHECKPOINT" \
  data.kwargs.toml_config_path="$TOML_CONFIG" \
  data.kwargs.perturbation_features_file="$ESM2_FEATURES" \
  data.kwargs.num_workers=$NUM_WORKERS \
  model.kwargs.input_dim=$INPUT_DIM \
  model.kwargs.hidden_dim=$HIDDEN_DIM \
  model.kwargs.output_dim=$OUTPUT_DIM \
  model.kwargs.pert_dim=$PERT_DIM \
  model.kwargs.se_model_path="$SE_MODEL_PATH" \
  model.kwargs.se_checkpoint_path="$SE_CHECKPOINT_PATH" \
  rl.env.reward_type="$REWARD_TYPE" \
  rl.training.n_episodes=$N_EPISODES \
  rl.training.learning_rate=$LEARNING_RATE \
  rl.training.batch_size=$BATCH_SIZE \
  output_dir="$OUTPUT_DIR" \
  name="$EXPERIMENT_NAME"

echo ""
echo "========================================="
echo "✅ RL Optimization completed!"
echo "========================================="
echo "Best model saved to: $OUTPUT_DIR/$EXPERIMENT_NAME/best_model.ckpt"
echo "Final model saved to: $OUTPUT_DIR/$EXPERIMENT_NAME/final_model.ckpt"
echo "Training stats: $OUTPUT_DIR/$EXPERIMENT_NAME/training_stats.pt"
echo "RL training finished at: $(date)"
echo ""
echo "The model has been fine-tuned using EXACT competition metrics:"
echo "  - DES (Differential Expression Score)"
echo "  - PDS (Perturbation Discrimination Score)"
echo "  - MAE (Mean Absolute Error)"
echo ""
echo "✅ You can now use these .ckpt files for inference!"
echo "   se-st-infer --checkpoint $OUTPUT_DIR/$EXPERIMENT_NAME/best_model.ckpt ..."
echo ""
