#!/bin/bash
# SE+ST Upgrade - RL on AdapterTune Model
# Apply RL optimization after adapter fine-tuning

set -e  # Exit on error

echo "========================================="
echo "SE+ST Upgrade - RL from AdapterTune"
echo "========================================="
echo ""

# Change to se_st_upgrade directory
cd /workspace/se_st_upgrade || {
    echo "Error: /workspace/se_st_upgrade not found"
    echo "Please adjust the path to your se_st_upgrade directory"
    exit 1
}

echo "Working directory: $(pwd)"
echo "Starting RL training on adapter model at: $(date)"
echo ""

# Model checkpoint (from AdapterTune)
MODEL_CHECKPOINT="competition/adaptertune/final_model.ckpt"

# Check if checkpoint exists
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "❌ Error: Adapter model checkpoint not found at $MODEL_CHECKPOINT"
    echo ""
    echo "Please run ./scripts/train_adaptertune.sh first"
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

# Model parameters
INPUT_DIM=18080
HIDDEN_DIM=512
OUTPUT_DIM=18080
PERT_DIM=5120
SE_MODEL_PATH="SE-600M"
SE_CHECKPOINT_PATH="SE-600M/se600m_epoch15.ckpt"

# Output configuration
OUTPUT_DIR="competition"
EXPERIMENT_NAME="rl_from_adapter"

echo "Configuration:"
echo "  Base model: $MODEL_CHECKPOINT (with adapters)"
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
echo "✅ RL optimization (on adapter model) completed!"
echo "========================================="
echo "Best model saved to: $OUTPUT_DIR/$EXPERIMENT_NAME/best_model.ckpt"
echo "Final model saved to: $OUTPUT_DIR/$EXPERIMENT_NAME/final_model.ckpt"
echo "RL training finished at: $(date)"
echo ""
echo "This is your FINAL optimized model:"
echo "  Base model → AdapterTune → RL optimization"
echo ""
echo "✅ You can now use these .ckpt files for inference!"
echo "   se-st-infer --checkpoint $OUTPUT_DIR/$EXPERIMENT_NAME/best_model.ckpt ..."
echo ""
