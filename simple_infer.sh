#!/bin/bash
# Simple inference script with all parameters inline

cd /workspace/se_st_upgrade || exit 1

export PYTHONPATH=/workspace/se_st_upgrade:$PYTHONPATH

echo "Running inference with 200K model..."

python -c "
import sys
sys.path.insert(0, '/workspace/se_st_upgrade')

# Set up arguments
sys.argv = [
    'infer',
    '--checkpoint', 'competition/baseline_200k/final_model.ckpt',
    '--adata', '/data/competition_val_template.h5ad',
    '--output', 'competition/baseline_200k_predictions_raw.h5ad',
    '--perturbation-features', '/data/ESM2_pert_features.pt',
    '--se-model-path', 'SE-600M',
    '--batch-size', '16',
    '--device', 'cuda'
]

from cli.infer import main
sys.exit(main())
"

echo "✅ Inference completed!"
ls -lh competition/baseline_200k_predictions_raw.h5ad
