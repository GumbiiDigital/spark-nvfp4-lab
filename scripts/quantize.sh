#!/usr/bin/env bash
# Quantize a HF model to NVFP4 using TensorRT-LLM Spark image.
# Usage: quantize.sh <model_id> <output_dir> <calib_dataset>
#   model_id        e.g. deepseek-ai/DeepSeek-R1-Distill-Llama-8B
#   output_dir      absolute host path; container will mount as /workspace/output_models
#   calib_dataset   "default" (cnn_dailymail) or "code" (codeparrot subset)
# Env: HF_TOKEN required.

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

MODEL_ID="${1:?missing model_id}"
OUT_DIR="${2:?missing output_dir absolute path}"
CALIB="${3:-default}"

require_hf_token
require_disk_gb "$HOME" 50

mkdir -p "$OUT_DIR"

CONTAINER_NAME="nvfp4-quant-$(date +%s)"
PHASE_LOG="$SPARK_LAB_ROOT/logs/quant-$(basename "$MODEL_ID")-$CALIB.log"

# Calibration dataset selection
case "$CALIB" in
  default) CALIB_OPT="" ;;  # script defaults to cnn_dailymail
  code)    CALIB_OPT="-e CALIB_DATASET=codeparrot/codeparrot-clean" ;;
  *)       error "unknown calib '$CALIB' (use default|code)"; exit 2 ;;
esac

event "QUANT_START" "model=$MODEL_ID calib=$CALIB out=$OUT_DIR"
heartbeat "quantize:$MODEL_ID:$CALIB"

# 4-hour budget for any single quantization run
with_timeout 14400 docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$OUT_DIR:/workspace/output_models" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e GPU_MAX_MEM_PERCENTAGE=0.9 \
  $CALIB_OPT \
  --name "$CONTAINER_NAME" \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c "
    git clone -b 0.35.0 --single-branch https://github.com/NVIDIA/Model-Optimizer.git /app/TensorRT-Model-Optimizer && \
    cd /app/TensorRT-Model-Optimizer && pip install -e '.[dev]' && \
    export ROOT_SAVE_PATH='/workspace/output_models' && \
    /app/TensorRT-Model-Optimizer/examples/llm_ptq/scripts/huggingface_example.sh \
    --model '$MODEL_ID' \
    --quant nvfp4 \
    --tp 1 \
    --export_fmt hf
  " >> "$PHASE_LOG" 2>&1

# Sanity: did we produce safetensors?
shards=$(find "$OUT_DIR" -name "*.safetensors" | wc -l)
if [ "$shards" -lt 1 ]; then
  error "quantize.sh produced 0 safetensor shards in $OUT_DIR"
  event "QUANT_FAIL" "model=$MODEL_ID no_shards"
  exit 3
fi

total_bytes=$(find "$OUT_DIR" -name "*.safetensors" -exec stat -c%s {} \; | awk '{s+=$1} END {print s}')
event "QUANT_OK" "model=$MODEL_ID calib=$CALIB shards=$shards size_bytes=$total_bytes"
info "quantize ok: $shards shards, $((total_bytes/1024/1024)) MB"
