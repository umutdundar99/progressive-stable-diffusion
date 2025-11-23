#!/bin/zsh
set -euo pipefail

python -m src.pipelines.training_pipeline "$@"
