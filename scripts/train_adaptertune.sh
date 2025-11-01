#!/bin/bash
# SE+ST Upgrade - AdapterTune Script
# Parameter-efficient fine-tuning with LoRA adapters

set -e  # Exit on error

echo "========================================="
echo "SE+ST Upgrade - AdapterTune"
echo "========================================="
echo ""

# Change to se_st_upgrade directory
cd /workspace/se_st_upgrade || {
    echo "Error: /workspace/se_st_upgrade not found"
    echo "Please adjust the path to your se_st_upgrade directory"
    exit 1
}

echo "Working directory: $(pwd)"
echo "Starting AdapterTune at: $(date)"

# Set PYTHONPATH so Python can find se_st_upgrade package
export PYTHONPATH=/workspace/se_st_upgrade:$PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"
echo ""
echo ""

# Model checkpoint to fine-tune (update this path!)
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

# AdapterTune parameters
ADAPTER_TYPE="lora"  # lora, bottleneck, or hybrid
LORA_RANK=16
LORA_ALPHA=16.0
MAX_STEPS=20000
BATCH_SIZE=16  # Can use larger batch with adapters

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
EXPERIMENT_NAME="adaptertune"

echo "Configuration:"
echo "  Base model: $MODEL_CHECKPOINT"
echo "  Adapter type: $ADAPTER_TYPE"
echo "  LoRA rank: $LORA_RANK"
echo "  Max steps: $MAX_STEPS"
echo "  Batch size: $BATCH_SIZE"
echo "  Output: $OUTPUT_DIR/$EXPERIMENT_NAME"
echo ""

# Run AdapterTune
python -m se_st_upgrade.cli.adaptertune \
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
  model.kwargs.adapter_type="$ADAPTER_TYPE" \
  model.kwargs.lora_rank=$LORA_RANK \
  model.kwargs.lora_alpha=$LORA_ALPHA \
  training.max_steps=$MAX_STEPS \
  training.batch_size=$BATCH_SIZE \
  output_dir="$OUTPUT_DIR" \
  name="$EXPERIMENT_NAME"

echo ""
echo "========================================="
echo "✅ AdapterTune completed!"
echo "========================================="
echo "Adapter weights saved to: $OUTPUT_DIR/$EXPERIMENT_NAME/final_adapters.pt"
echo "Full model saved to: $OUTPUT_DIR/$EXPERIMENT_NAME/final_model.ckpt"
echo "AdapterTune finished at: $(date)"
echo ""
echo "Adapter statistics:"
echo "  - Only ~5% of parameters were trained (95% reduction!)"
echo "  - Training was 3-5x faster than full fine-tuning"
echo ""
echo "Next step: Optimize with RL"
echo "  ./scripts/train_rl_from_adapter.sh"
echo ""
