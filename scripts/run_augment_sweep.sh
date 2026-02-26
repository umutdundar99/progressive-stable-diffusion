#!/usr/bin/env bash
# Generate balanced datasets for multiple steer scales in sequence.
# Usage: bash scripts/run_augment_sweep.sh

set -euo pipefail
cd /home/umut_dundar/repositories/progressive-stable-diffusion
export PYTHONPATH="$PWD"

CHECKPOINT="checkpoints/prog-disease-generation-ip-final/icmmf2dn_no_routing_no_purifier/checkpoints/ip-ddpm-epochepoch=0149.ckpt"
CONFIG="configs/train_ip.yaml"
DATA_ROOT="data/limuc_cleaned"

STEER_SCALES=(1.5 2.0 2.5 3.0)
SUFFIXES=(15 20 25 30)

for i in "${!STEER_SCALES[@]}"; do
    scale="${STEER_SCALES[$i]}"
    suffix="${SUFFIXES[$i]}"
    output="data/icmmf2dn/limuc_cleaned_balanced_icmmf2dn_steer_${suffix}"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Steer scale: ${scale}  →  ${output}"
    echo "════════════════════════════════════════════════════════════"

    python src/pipelines/inference/inference_pipeline_ip_data_augment.py \
        --checkpoint "$CHECKPOINT" \
        --config "$CONFIG" \
        --data-root "$DATA_ROOT" \
        --output-root "$output" \
        --image-scale 1 \
        --guidance-scale "$scale" \
        --seed 42 \
        --batch-images 4 \
        --save-workers 4

    echo "✅ Done: guidance_scale=${scale}"
done

echo ""
echo "🎉 All guidance scale sweeps completed!"
