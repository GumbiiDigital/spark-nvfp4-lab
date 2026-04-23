#!/usr/bin/env bash
# Spin up an ephemeral trtllm-serve container against a quantized (or BF16) model
# Used by eval.sh and perf.sh — eval clients hit the OpenAI-compatible endpoint.
# Usage:
#   ./serve.sh start <model_path_on_host> <name> <port>
#   ./serve.sh stop  <name>
#   ./serve.sh wait  <port>     # block until /v1/models returns 200
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

CMD="${1:?missing subcommand: start|stop|wait}"

case "$CMD" in
  start)
    MODEL_PATH="${2:?missing model path}"; NAME="${3:?missing container name}"; PORT="${4:?missing port}"
    LOG="$SPARK_LAB_ROOT/logs/serve-$NAME.log"
    info "starting trtllm-serve name=$NAME port=$PORT model=$MODEL_PATH"
    nohup docker run --rm --gpus all --ipc=host --network host \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      -e HF_TOKEN="${HF_TOKEN:-}" \
      -v "$MODEL_PATH:/workspace/model" \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      --name "$NAME" \
      nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
      trtllm-serve /workspace/model \
        --backend pytorch \
        --max_batch_size 4 \
        --port "$PORT" \
      > "$LOG" 2>&1 &
    disown
    info "trtllm-serve dispatched, log=$LOG"
    ;;
  stop)
    NAME="${2:?missing container name}"
    info "stopping container $NAME"
    docker stop -t 10 "$NAME" 2>/dev/null || true
    docker rm -f "$NAME" 2>/dev/null || true
    ;;
  wait)
    PORT="${2:?missing port}"
    info "waiting for :$PORT /v1/models (10 min budget)"
    for i in $(seq 1 60); do
      if curl -s -m 3 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q 200; then
        info "trtllm-serve :$PORT ready (~$((i*10))s)"
        exit 0
      fi
      sleep 10
    done
    error "trtllm-serve :$PORT never became ready"
    exit 1
    ;;
  *) error "unknown subcommand: $CMD"; exit 2 ;;
esac
