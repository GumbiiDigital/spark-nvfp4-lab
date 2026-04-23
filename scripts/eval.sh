#!/usr/bin/env bash
# Run lm-evaluation-harness against an OpenAI-compatible endpoint.
# Generation-task only (gsm8k, humaneval) — logprob tasks (mmlu, wikitext)
# can't run via the API path because llama-server / trtllm-serve don't expose
# token_logprobs in the legacy OpenAI shape lm-eval requires.
#
# Usage: eval.sh <port> <out_dir> <tokenizer> [task_set]
#   port        OpenAI-compat endpoint host port
#   out_dir     directory to write per-task JSON results
#   tokenizer   HF tokenizer id (e.g. deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
#   task_set    v0 (default: gsm8k+humaneval)
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

PORT="${1:?missing port}"
OUT_DIR="${2:?missing out_dir}"
TOKENIZER="${3:?missing tokenizer (e.g. meta-llama/Llama-3.1-8B-Instruct)}"
TASKS_KEY="${4:-v0}"
mkdir -p "$OUT_DIR"

case "$TASKS_KEY" in
  v0)   TASKS="gsm8k" ;;          # generation-only — works via API
  full) TASKS="gsm8k" ;;          # same for now; v0.2 expands once direct loader lands
  *) error "unknown task_set: $TASKS_KEY"; exit 2 ;;
esac

event "EVAL_START" "port=$PORT out=$OUT_DIR tasks=$TASKS"
heartbeat "eval:$PORT:$TASKS_KEY"

# Endpoint sanity
if ! curl -s -m 5 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q 200; then
  error "endpoint :$PORT not responding /v1/models"
  exit 1
fi

LMEVAL_CONTAINER="lm-eval-runner-$(date +%s)"
LMEVAL_LOG="$SPARK_LAB_ROOT/logs/eval-$(basename "$OUT_DIR")-$TASKS_KEY.log"

# 6-hour budget across the v0 task set
with_timeout 21600 docker run --rm --network host \
  --name "$LMEVAL_CONTAINER" \
  -v "$OUT_DIR:/results" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c "
    pip install -q lm-eval[api]==0.4.7 2>&1 | tail -1 && \
    lm-eval \
      --model local-completions \
      --model_args base_url=http://127.0.0.1:$PORT/v1/completions,model=quant,tokenizer=$TOKENIZER,tokenizer_backend=huggingface,tokenized_requests=False,num_concurrent=1 \
      --tasks $TASKS \
      --num_fewshot 5 \
      --batch_size 1 \
      --output_path /results
  " >> "$LMEVAL_LOG" 2>&1 || {
    error "lm-eval failed (see $LMEVAL_LOG)"
    event "EVAL_FAIL" "port=$PORT tasks=$TASKS"
    exit 3
  }

# lm-eval writes results to a subdir like /results/<model>/results_<ts>.json
result_json=$(find "$OUT_DIR" -name "results_*.json" -print -quit)
if [ -z "$result_json" ]; then
  error "no results_*.json produced"
  exit 4
fi
cp "$result_json" "$OUT_DIR/results.json"
event "EVAL_OK" "port=$PORT tasks=$TASKS_KEY result=$OUT_DIR/results.json"
info "eval complete: $result_json -> $OUT_DIR/results.json"
