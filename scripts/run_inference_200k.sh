#!/bin/bash
# SE+ST Upgrade - Run Inference with 200K Baseline Model
# Generate competition submission predictions

set -e  # Exit on error

echo "========================================="
echo "SE+ST Upgrade - Inference (200K Model)"
echo "========================================="
echo ""

# Change to se_st_upgrade directory
cd /workspace/se_st_upgrade || {
    echo "Error: /workspace/se_st_upgrade not found"
    exit 1
}

echo "Working directory: $(pwd)"
echo "Starting inference at: $(date)"
echo ""

# Set PYTHONPATH
export PYTHONPATH=/workspace/se_st_upgrade:$PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"
echo ""

# Model checkpoint
CHECKPOINT="competition/baseline_200k/final_model.ckpt"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found at $CHECKPOINT"
    echo ""
    echo "Please ensure baseline_200k training completed successfully"
    exit 1
fi

# Input/Output paths
INPUT_ADATA="/data/competition_val_template.h5ad"
OUTPUT_ADATA="competition/baseline_200k_predictions.h5ad"
PERT_FEATURES="/data/ESM2_pert_features.pt"
SE_MODEL_PATH="SE-600M"

# Inference parameters
BATCH_SIZE=16
DEVICE="cuda"

echo "Configuration:"
echo "  📦 Model checkpoint: $CHECKPOINT"
echo "  📊 Input data: $INPUT_ADATA"
echo "  💾 Output predictions: $OUTPUT_ADATA"
echo "  🧬 Perturbation features: $PERT_FEATURES"
echo "  🖥️  Device: $DEVICE"
echo "  📏 Batch size: $BATCH_SIZE"
echo ""

# Run inference using the CLI wrapper
python run_infer.py \
  --checkpoint "$CHECKPOINT" \
  --adata "$INPUT_ADATA" \
  --output "$OUTPUT_ADATA" \
  --pert-col target_gene \
  --se-model-path "$SE_MODEL_PATH" \
  --perturbation-features "$PERT_FEATURES" \
  --batch-size $BATCH_SIZE \
  --device $DEVICE

echo ""
echo "========================================="
echo "✅ Inference completed!"
echo "========================================="
echo "Predictions saved to: $OUTPUT_ADATA"
echo "Inference finished at: $(date)"
echo ""
echo "📊 Check output:"
echo "  python -c \"import anndata as ad; adata = ad.read_h5ad('$OUTPUT_ADATA'); print(f'Shape: {adata.shape}'); print(f'Sparse: {type(adata.X)}')\""
echo ""
echo "📤 Next steps:"
echo "  1. Verify output format matches competition requirements"
echo "  2. Submit predictions to competition"
echo "  3. Compare with baseline scores"
echo ""
