#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m src.web.app --host "${HOST:-127.0.0.1}" --port "${PORT:-7860}"
