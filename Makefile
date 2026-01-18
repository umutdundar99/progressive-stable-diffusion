SHELL := /bin/zsh
UV := uv
PY := python

.PHONY: help format lint train inference

help:
	@echo "Available targets: format, lint, train, inference"

format:
	$(UV) format

lint:
	$(UV) check

train:
	$(UV) run scripts/run_train.sh

inference:
	$(UV) run scripts/run_inference.sh

# Compare two IP-Adapter checkpoints with various guidance scales
compare-ip:
	chmod +x scripts/compare_ip_checkpoints.sh
	./scripts/compare_ip_checkpoints.sh

# Quick comparison (fewer samples and guidance scales)
compare-ip-quick:
	chmod +x scripts/compare_ip_checkpoints.sh
	./scripts/compare_ip_checkpoints.sh --quick

# Compare with custom checkpoints
# Usage: make compare-ip-custom CKPT1=path/to/ckpt1 CKPT2=path/to/ckpt2
compare-ip-custom:
	chmod +x scripts/compare_ip_checkpoints.sh
	./scripts/compare_ip_checkpoints.sh --checkpoint1 $(CKPT1) --checkpoint2 $(CKPT2)
