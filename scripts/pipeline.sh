#!/usr/bin/env bash
# Spark NVFP4 Lab — main pipeline orchestrator with guardrails.
#
# Usage: pipeline.sh <release_tag> <model_id> [calibration]
#   release_tag    e.g. release-0_DeepSeek-R1-Distill-Llama-8B
#   model_id       HF model id, e.g. deepseek-ai/DeepSeek-R1-Distill-Llama-8B
#   calibration    default | code  (default: default)
#
# Runs: snapshot services -> stop services -> quantize -> serve quant ->
#       eval NVFP4 -> stop serve -> serve BF16 -> eval BF16 -> stop serve ->
#       perf (NVFP4) -> render card -> restart services -> done
#
# Quality gates are enforced via render_card.py exit code 10 (auto-publish blocked).
# Publishing is NEVER automatic — a human reviews and runs publish.sh.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

RELEASE_TAG="${1:?missing release_tag}"
MODEL_ID="${2:?missing model_id}"
CALIB="${3:-default}"

acquire_lock
require_hf_token
require_disk_gb "$HOME" 50

RELEASE_DIR="$SPARK_LAB_ROOT/results/$RELEASE_TAG"
mkdir -p "$RELEASE_DIR"/{quant,eval-bf16,eval-nvfp4,perf}

# State file — track which phases are done so we can resume after crash
STATE_FILE="$RELEASE_DIR/state.json"
[ -f "$STATE_FILE" ] || echo '{"phases": {}}' > "$STATE_FILE"

phase_done() { jq -r ".phases.\"$1\" // empty" "$STATE_FILE" | grep -q "^ok$"; }
mark_phase() {
  local p="$1"; local s="$2"
  jq ".phases.\"$p\" = \"$s\"" "$STATE_FILE" > "$STATE_FILE.tmp"
  mv "$STATE_FILE.tmp" "$STATE_FILE"
}

# Heartbeat daemon — ticks every 5 min so the user can `cat heartbeat.txt`
( while true; do heartbeat "running:$RELEASE_TAG"; sleep 300; done ) &
HEARTBEAT_PID=$!
trap 'kill $HEARTBEAT_PID 2>/dev/null; release_lock' EXIT

event "PIPELINE_START" "release=$RELEASE_TAG model=$MODEL_ID calib=$CALIB"

# ---- Phase: snapshot + stop services ----------------------------------------
if ! phase_done snapshot_services; then
  info "phase: snapshot services"
  snapshot_services
  cp "$SPARK_LAB_SERVICE_STATE" "$RELEASE_DIR/services_snapshot.json"
  mark_phase snapshot_services ok
fi
if ! phase_done stop_services; then
  info "phase: stop services"
  stop_services
  sleep 5
  mark_phase stop_services ok
fi

# ---- Phase: quantize --------------------------------------------------------
if ! phase_done quantize; then
  info "phase: quantize"
  "$SCRIPT_DIR/quantize.sh" "$MODEL_ID" "$RELEASE_DIR/quant" "$CALIB"
  # Write quant manifest
  shards=$(find "$RELEASE_DIR/quant" -name "*.safetensors")
  total=$(echo "$shards" | xargs -I{} stat -c%s {} 2>/dev/null | awk '{s+=$1} END {print s}')
  jq -n \
    --arg model "$MODEL_ID" \
    --arg calib "$CALIB" \
    --arg ts "$(ts)" \
    --argjson size "$total" \
    --argjson nshards "$(echo "$shards" | grep -c safetensors)" \
    '{model: $model, calibration: $calib, generated_utc: $ts, total_bytes: $size, n_shards: $nshards}' \
    > "$RELEASE_DIR/quant/manifest.json"
  mark_phase quantize ok
else
  info "phase: quantize (skipped — already done)"
fi

# Determine the actual quant subdirectory (TRT-LLM creates saved_models_X/).
# -L follows symlinks — release-0 uses a symlink into ~/nvfp4/output_models.
QUANT_OUT_DIR=$(find -L "$RELEASE_DIR/quant" -maxdepth 1 -mindepth 1 -type d -name "saved_models_*" | head -1)
if [ -z "$QUANT_OUT_DIR" ]; then
  error "no saved_models_* directory in $RELEASE_DIR/quant"
  exit 1
fi
info "quantized weights at: $QUANT_OUT_DIR"

# ---- Phase: eval NVFP4 ------------------------------------------------------
if ! phase_done eval_nvfp4; then
  info "phase: serve NVFP4 + eval"
  "$SCRIPT_DIR/serve.sh" start "$QUANT_OUT_DIR" "trtllm-nvfp4-$$" 8000
  if ! "$SCRIPT_DIR/serve.sh" wait 8000; then
    "$SCRIPT_DIR/serve.sh" stop "trtllm-nvfp4-$$" || true
    error "NVFP4 serve never came up"
    exit 1
  fi
  "$SCRIPT_DIR/eval.sh" 8000 "$RELEASE_DIR/eval-nvfp4" "$MODEL_ID" v0
  "$SCRIPT_DIR/serve.sh" stop "trtllm-nvfp4-$$"
  sleep 5
  mark_phase eval_nvfp4 ok
else
  info "phase: eval_nvfp4 (skipped)"
fi

# ---- Phase: perf ------------------------------------------------------------
if ! phase_done perf; then
  info "phase: serve NVFP4 + perf"
  "$SCRIPT_DIR/serve.sh" start "$QUANT_OUT_DIR" "trtllm-perf-$$" 8000
  if ! "$SCRIPT_DIR/serve.sh" wait 8000; then
    "$SCRIPT_DIR/serve.sh" stop "trtllm-perf-$$" || true
    error "NVFP4 serve for perf never came up"
    exit 1
  fi
  "$SCRIPT_DIR/perf.sh" 8000 "$RELEASE_DIR/perf/perf.json"
  "$SCRIPT_DIR/serve.sh" stop "trtllm-perf-$$"
  sleep 5
  mark_phase perf ok
else
  info "phase: perf (skipped)"
fi

# ---- Phase: eval BF16 baseline ---------------------------------------------
# Skipped if already done. Uses HF transformers via lm-eval --model hf.
if ! phase_done eval_bf16; then
  info "phase: eval BF16 baseline (HF transformers backend)"
  BF16_LOG="$SPARK_LAB_ROOT/logs/eval-bf16-$RELEASE_TAG.log"
  with_timeout 21600 docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --name "lm-eval-bf16-$$" \
    -v "$RELEASE_DIR/eval-bf16:/results" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e HF_TOKEN="$HF_TOKEN" \
    nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
    bash -c "
      pip install -q lm-eval==0.4.7 2>&1 | tail -3 && \
      lm-eval \
        --model hf \
        --model_args pretrained=$MODEL_ID,dtype=bfloat16,trust_remote_code=True \
        --tasks mmlu,gsm8k,wikitext \
        --batch_size auto \
        --output_path /results
    " >> "$BF16_LOG" 2>&1
  result=$(find "$RELEASE_DIR/eval-bf16" -name "results_*.json" -print -quit)
  [ -n "$result" ] && cp "$result" "$RELEASE_DIR/eval-bf16/results.json"
  mark_phase eval_bf16 ok
else
  info "phase: eval_bf16 (skipped)"
fi

# ---- Phase: render card -----------------------------------------------------
info "phase: render card + manifest + REPRODUCE.sh"
if python3 "$SCRIPT_DIR/render_card.py" \
     --release-dir "$RELEASE_DIR" \
     --model-id "$MODEL_ID" \
     --calibration "$CALIB"; then
  mark_phase render ok
  event "RENDER_OK" "release=$RELEASE_TAG"
else
  rc=$?
  if [ "$rc" -eq 10 ]; then
    warn "quality gates triggered — auto-publish blocked, manual review required"
    mark_phase render gated
    event "RENDER_GATED" "release=$RELEASE_TAG"
  else
    error "render_card.py exited $rc"
    mark_phase render fail
    exit "$rc"
  fi
fi

# ---- Phase: restart services -----------------------------------------------
if ! phase_done restore_services; then
  info "phase: restore services"
  restore_services
  sleep 10
  # Verify ports — best effort, log warnings only
  for p in 8081 8082 8765 8766 8880; do
    if ss -tlnp 2>/dev/null | grep -q ":$p "; then
      info "service port :$p restored"
    else
      warn "service port :$p NOT restored — check /tmp/$p.restored.log"
    fi
  done
  mark_phase restore_services ok
fi

event "PIPELINE_OK" "release=$RELEASE_TAG dir=$RELEASE_DIR"
info "pipeline complete: $RELEASE_DIR"
info "review: cat $RELEASE_DIR/BENCHMARKS.md"
info "if good: $SCRIPT_DIR/publish.sh $RELEASE_TAG  (manual gate)"
