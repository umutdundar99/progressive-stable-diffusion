#!/usr/bin/env bash
# Generate balanced datasets for multiple steer scales in sequence.
# Usage: bash scripts/run_augment_sweep.sh

set -euo pipefail
cd /home/umut_dundar/repositories/progressive-stable-diffusion
export PYTHONPATH="$PWD"

CHECKPOINT="prog-disease-generation-ip/uqqx9kg9/checkpoints/ip-ddpm-epochepoch=0149.ckpt"
CONFIG="configs/train_ip.yaml"
DATA_ROOT="data/limuc_cleaned"

STEER_SCALES=(1.0 1.5 2.0 2.5 3.0)
SUFFIXES=(10 15 20 25 30)

for i in "${!STEER_SCALES[@]}"; do
    scale="${STEER_SCALES[$i]}"
    suffix="${SUFFIXES[$i]}"
    output="data/limuc_cleaned_balanced_uqqx9kg9_steer_${suffix}"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Steer scale: ${scale}  →  ${output}"
    echo "════════════════════════════════════════════════════════════"

    python src/pipelines/inference/inference_pipeline_ip_data_augment.py \
        --checkpoint "$CHECKPOINT" \
        --config "$CONFIG" \
        --data-root "$DATA_ROOT" \
        --output-root "$output" \
        --sampling-steps 50 \
        --guidance-scale 1 \
        --image-scale 1 \
        --steer-scale "$scale" \
        --no-blur \
        --seed 42 \
        --batch-images 8 \
        --save-workers 4

    echo "✅ Done: steer_scale=${scale}"
done

echo ""
echo "🎉 All steer scale sweeps completed!"
