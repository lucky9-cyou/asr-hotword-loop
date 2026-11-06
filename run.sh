#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/.deps${PYTHONPATH:+:$PYTHONPATH}"
export ASR_DEVICE="${ASR_DEVICE:-cuda:0}"
export LLM_DEVICE="${LLM_DEVICE:-cuda:1}"
export LLM_MODEL="${LLM_MODEL:-/workspace/model/Qwen3.8-27B}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
exec /workspace/workspace/yanglin/envs/lm-tts/bin/python -m uvicorn server:app --host "${HOST:-0.0.0.0}" --port "${PORT:-7860}" --timeout-graceful-shutdown 5
