#!/bin/bash
set -e

NVFP4_MODEL=$HOME/spark-lab/results/release-0_DeepSeek-R1-Distill-Llama-8B/quant/saved_models_DeepSeek-R1-Distill-Llama-8B_nvfp4_hf
BF16_MODEL=$HOME/spark-lab/models/DeepSeek-R1-Distill-Llama-8B-bf16
RELEASE=$HOME/spark-lab/results/release-0_DeepSeek-R1-Distill-Llama-8B

run_humaneval() {
    local MODEL_DIR=$1
    local TAG=$2
    local OUT_DIR=$3
    local NAME="trtllm-he-${TAG}-$$"

    echo "=== [$TAG] $(date -u +%H:%M:%S) starting serve ==="
    ~/spark-lab/scripts/serve.sh start "$MODEL_DIR" "$NAME" 8000
    ~/spark-lab/scripts/serve.sh wait 8000

    echo "=== [$TAG] $(date -u +%H:%M:%S) running humaneval ==="
    mkdir -p "$OUT_DIR/humaneval-run"
    docker run --rm --network host \
        --name "he-runner-${TAG}-$$" \
        -v "$OUT_DIR/humaneval-run:/results" \
        -e HF_ALLOW_CODE_EVAL=1 \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
        bash -c "
            pip install -q lm-eval[api]==0.4.7 2>&1 | tail -1 && \
            lm-eval \
                --model local-completions \
                --model_args base_url=http://127.0.0.1:8000/v1/completions,model=quant,tokenizer=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,tokenizer_backend=huggingface,tokenized_requests=False,num_concurrent=4 \
                --tasks humaneval \
                --batch_size 1 \
                --output_path /results
        "

    echo "=== [$TAG] $(date -u +%H:%M:%S) stopping serve ==="
    ~/spark-lab/scripts/serve.sh stop "$NAME"
}

run_humaneval "$NVFP4_MODEL" "nvfp4" "$RELEASE/eval-nvfp4"
run_humaneval "$BF16_MODEL"  "bf16"  "$RELEASE/eval-bf16"

echo "=== $(date -u +%H:%M:%S) DONE ==="
