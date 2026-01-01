#!/bin/bash

# 1. Get the seed number from the command line argument
SEED=$1

# Check if a seed was provided
if [ -z "$SEED" ]; then
  echo "Error: Please provide a seed number."
  echo "Usage: ./run_inference.sh <seed_number>"
  exit 1
fi


CKPT_BASE_PATH="/home/umut_dundar/repositories/progressive-stable-diffusion/prog-disease-generation/c8o3zs2i/checkpoints"

for i in $(seq 89 10 99); do

    CKPT_NUM_FORMATTED=$(printf "%04d" $i)

    CKPT_FILE="${CKPT_BASE_PATH}/ddpm-epochepoch=${CKPT_NUM_FORMATTED}.ckpt"

    OUTPUT_DIR="outputs2/inference_run_${SEED}_${i}"

    echo "----------------------------------------------------------------"
    echo "Starting Process -> Seed: $SEED | Checkpoint: $i"
    echo "Checkpoint File: $CKPT_FILE"
    echo "Output Dir: $OUTPUT_DIR"
    echo "----------------------------------------------------------------"

    # Run the python command
    python -m src.pipelines.inference_pipeline \
        --checkpoint "$CKPT_FILE" \
        --config configs/train.yaml \
        --sampling-steps 50 \
        --mes-steps 13 \
        --output-dir "$OUTPUT_DIR" \
        --seed "$SEED"

done

echo "All inference runs completed successfully."
