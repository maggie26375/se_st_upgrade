#!/bin/bash
# SE+ST Upgrade - AutoTune Script
# Automated hyperparameter search with Optuna

set -e  # Exit on error

echo "========================================="
echo "SE+ST Upgrade - AutoTune"
echo "========================================="
echo ""

# Change to se_st_upgrade directory
cd /workspace/se_st_upgrade || {
    echo "Error: /workspace/se_st_upgrade not found"
    echo "Please adjust the path to your se_st_upgrade directory"
    exit 1
}

echo "Working directory: $(pwd)"
echo "Starting AutoTune at: $(date)"
echo ""

# AutoTune parameters
N_TRIALS=50  # Number of hyperparameter configurations to try
MAX_STEPS_PER_TRIAL=10000  # Steps per trial (for quick evaluation)

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
EXPERIMENT_NAME="autotune_v1"

echo "Configuration:"
echo "  Trials: $N_TRIALS"
echo "  Steps per trial: $MAX_STEPS_PER_TRIAL"
echo "  Data config: $TOML_CONFIG"
echo "  Output: $OUTPUT_DIR/$EXPERIMENT_NAME"
echo ""
echo "⚠️  This will take approximately $(($N_TRIALS * 2)) hours"
echo "    (assuming ~2 hours per trial on A100)"
echo ""

# Run AutoTune
python -m cli.autotune \
  data.kwargs.toml_config_path="$TOML_CONFIG" \
  data.kwargs.perturbation_features_file="$ESM2_FEATURES" \
  data.kwargs.num_workers=$NUM_WORKERS \
  model.kwargs.input_dim=$INPUT_DIM \
  model.kwargs.hidden_dim=$HIDDEN_DIM \
  model.kwargs.output_dim=$OUTPUT_DIM \
  model.kwargs.pert_dim=$PERT_DIM \
  model.kwargs.se_model_path="$SE_MODEL_PATH" \
  model.kwargs.se_checkpoint_path="$SE_CHECKPOINT_PATH" \
  autotune.n_trials=$N_TRIALS \
  autotune.resources.max_steps=$MAX_STEPS_PER_TRIAL \
  output_dir="$OUTPUT_DIR" \
  name="$EXPERIMENT_NAME"

echo ""
echo "========================================="
echo "✅ AutoTune completed!"
echo "========================================="
echo "Best configuration saved to: $OUTPUT_DIR/$EXPERIMENT_NAME/best_config.yaml"
echo "Best parameters: $OUTPUT_DIR/$EXPERIMENT_NAME/best_params.json"
echo "Study summary: $OUTPUT_DIR/$EXPERIMENT_NAME/study_summary.json"
echo "Visualization: $OUTPUT_DIR/$EXPERIMENT_NAME/optimization_history.html"
echo "AutoTune finished at: $(date)"
echo ""
echo "Next step: Train with best configuration"
echo "  python -m cli.train \\"
echo "    --config-path $OUTPUT_DIR/$EXPERIMENT_NAME \\"
echo "    --config-name best_config \\"
echo "    training.max_steps=80000"
echo ""
