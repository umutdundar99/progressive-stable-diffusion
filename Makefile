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
