#!/bin/bash
# =====================================================================
# IP-Adapter Training Script
# =====================================================================
#
# This script trains the diffusion model with dual conditioning:
# - AOE (Additive Ordinal Embedding) for disease severity
# - Image features for patient-specific anatomical structure
#
# Usage: bash scripts/run_train_ip.sh
# =====================================================================

set -e

cd /home/umut_dundar/repositories/progressive-stable-diffusion

echo "============================================================"
echo "🚀 Starting IP-Adapter Training"
echo "============================================================"

# Run training
python -m src.pipelines.training_pipeline_ip

echo "============================================================"
echo "✅ IP-Adapter Training Complete!"
echo "============================================================"
