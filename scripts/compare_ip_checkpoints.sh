#!/bin/bash
# Script to compare IP-Adapter model checkpoints
#
# Model configurations:
# - kwd2qy49: blur=ON,  dominant=1.0, non_dominant=1.0 (uniform + blur)
# - pvq7gpe7: blur=ON,  dominant=1.5, non_dominant=0.5 (frequency-aware + blur)
# - ejfpqk2s: blur=OFF, dominant=1.0, non_dominant=1.0 (uniform + no blur)

# Default values
CHECKPOINT1="/home/umut_dundar/repositories/progressive-stable-diffusion/prog-disease-generation-ip/kwd2qy49/checkpoints/ip-ddpm-epochepoch=0149.ckpt"
CHECKPOINT2="/home/umut_dundar/repositories/progressive-stable-diffusion/prog-disease-generation-ip/pvq7gpe7/checkpoints/ip-ddpm-epochepoch=0149.ckpt"
CHECKPOINT3="/home/umut_dundar/repositories/progressive-stable-diffusion/prog-disease-generation-ip/ejfpqk2s/checkpoints/ip-ddpm-epochepoch=0149.ckpt"
NAME1="kwd2qy49_blur_uniform"
NAME2="pvq7gpe7_blur_freqaware"
NAME3="ejfpqk2s_noblur_uniform"
CONFIG="configs/train_ip.yaml"
REAL_DATA="/home/umut_dundar/repositories/progressive-stable-diffusion/data/limuc/processed_data_scale1/test"
OUTPUT_DIR="outputs/evaluation_ip_compare"
GUIDANCE_SCALES="0.0 0.5 1.0 1.5 2.0 3.0 5.0 7.5"
NUM_SAMPLES=50
SAMPLING_STEPS=50
BATCH_SIZE=8
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint1)
            CHECKPOINT1="$2"
            shift 2
            ;;
        --checkpoint2)
            CHECKPOINT2="$2"
            shift 2
            ;;
        --name1)
            NAME1="$2"
            shift 2
            ;;
        --name2)
            NAME2="$2"
            shift 2
            ;;
        --checkpoint3)
            CHECKPOINT3="$2"
            shift 2
            ;;
        --name3)
            NAME3="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --quick)
            GUIDANCE_SCALES="0.0 1.0 2.0"
            NUM_SAMPLES=10
            shift
            ;;
        --save-images)
            SAVE_IMAGES="--save-images"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "IP-Adapter Checkpoint Comparison"
echo "========================================"
echo "Model 1: $NAME1"
echo "  Checkpoint: $CHECKPOINT1"
echo "  Config: blur=ON, dominant=1.0, non_dominant=1.0"
echo ""
echo "Model 2: $NAME2"
echo "  Checkpoint: $CHECKPOINT2"
echo "  Config: blur=ON, dominant=1.5, non_dominant=0.5"
echo ""
echo "Model 3: $NAME3"
echo "  Checkpoint: $CHECKPOINT3"
echo "  Config: blur=OFF, dominant=1.0, non_dominant=1.0"
echo ""
echo "Guidance scales: $GUIDANCE_SCALES"
echo "Samples per class: $NUM_SAMPLES"
echo "========================================"

python -m src.pipelines.evaluation_pipeline_ip_compare \
    --checkpoint1 "$CHECKPOINT1" \
    --checkpoint2 "$CHECKPOINT2" \
    --checkpoint3 "$CHECKPOINT3" \
    --name1 "$NAME1" \
    --name2 "$NAME2" \
    --name3 "$NAME3" \
    --config "$CONFIG" \
    --real-data-dir "$REAL_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --guidance-scales $GUIDANCE_SCALES \
    --num-samples-per-class "$NUM_SAMPLES" \
    --sampling-steps "$SAMPLING_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    $SAVE_IMAGES
