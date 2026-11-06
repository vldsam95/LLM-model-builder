#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m src.training.train_pretrain --config configs/tiny_cpu.yaml
