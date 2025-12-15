#!/bin/zsh
set -euo pipefail

# Example evaluation script
# Usage: ./scripts/run_evaluation.sh

python -m src.pipelines.evaluation_pipeline \
    --checkpoint /path/to/checkpoint.ckpt \
    --config configs/train.yaml \
    --real-data-dir /path/to/real/images \
    --output-dir outputs/evaluation \
    --num-samples-per-class 100 \
    --sampling-steps 50 \
    --sampler ddim \
    --batch-size 16 \
    --seed 42 \
    --save-images \
    "$@"
