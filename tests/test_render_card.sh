#!/usr/bin/env bash
# Functional test for render_card.py. Asserts on the OUTPUT of a known-input run.
# Not a full golden-file diff (too much dynamic content: timestamps, sha of the
# script, container digest from local docker, etc.) — instead we pin the invariants
# that matter: real metric numbers land in the right places, no template leaks,
# YAML frontmatter is valid, compression math is right.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FIXTURE="$SCRIPT_DIR/fixtures/release-sample"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Copy fixture to a writable scratch location (render_card writes README etc into it)
cp -r "$FIXTURE/." "$WORK/"

# Drive the renderer — use --override-gate-exit because the fixture triggers the
# 3pp math gate (same real -3.94pp delta as release-0).
python3 "$REPO_ROOT/scripts/render_card.py" \
  --release-dir "$WORK" \
  --model-id "fixture/SampleModel" \
  --calibration default \
  --license mit \
  --eval-context-length 2048 \
  --override-gate-exit "CI fixture — expected to trip the 3pp gate"

pass=0
fail=0

check() {
  local desc="$1" expected="$2" file="$3"
  if grep -qF -- "$expected" "$file" 2>/dev/null; then
    echo "  PASS: $desc"
    pass=$((pass+1))
  else
    echo "  FAIL: $desc (looking for '$expected' in $file)"
    fail=$((fail+1))
  fi
}

absent() {
  local desc="$1" pattern="$2" file="$3"
  if grep -qF -- "$pattern" "$file" 2>/dev/null; then
    echo "  FAIL: $desc ('$pattern' found in $file but should be absent)"
    fail=$((fail+1))
  else
    echo "  PASS: $desc"
    pass=$((pass+1))
  fi
}

jq_eq() {
  local desc="$1" jq_expr="$2" expected="$3" file="$4"
  local got
  got=$(jq -r "$jq_expr" "$file" 2>/dev/null || echo "")
  if [ "$got" = "$expected" ]; then
    echo "  PASS: $desc (got '$got')"
    pass=$((pass+1))
  else
    echo "  FAIL: $desc (expected '$expected', got '$got')"
    fail=$((fail+1))
  fi
}

echo "=== README.md ==="
check "YAML frontmatter opens" "---" "$WORK/README.md"
check "model id rendered in title" "SampleModel" "$WORK/README.md"
check "license field populated" "license: mit" "$WORK/README.md"
check "explicit nvfp4 tag present" "- nvfp4" "$WORK/README.md"
check "4-bit tag present" "- 4-bit" "$WORK/README.md"
absent "no leftover {args.} template leaks" "{args." "$WORK/README.md"
absent "no see-original license leak" "license: see-original" "$WORK/README.md"
absent "8-bit tag NOT present" "- 8-bit" "$WORK/README.md"
check "headline includes delta number" "-3.94 pp" "$WORK/README.md"
check "known-good trtllm-serve snippet" "trtllm-serve /workspace/model" "$WORK/README.md"

echo "=== BENCHMARKS.md ==="
check "BF16 gsm8k value rendered" "0.6452" "$WORK/BENCHMARKS.md"
check "NVFP4 gsm8k value rendered" "0.6058" "$WORK/BENCHMARKS.md"
check "delta column populated" "-3.94" "$WORK/BENCHMARKS.md"
check "perf interp picks c=4 saturation" "saturates at concurrency ≈ 4" "$WORK/BENCHMARKS.md"
check "context-length caveat present" "Caveat — context length" "$WORK/BENCHMARKS.md"
check "skipped-tasks block present" "Tasks **not** evaluated" "$WORK/BENCHMARKS.md"

echo "=== manifest.json ==="
jq_eq "BF16 gsm8k score pinned"  ".results.bf16.gsm8k.value"  "0.6452"                    "$WORK/manifest.json"
jq_eq "NVFP4 gsm8k score pinned" ".results.nvfp4.gsm8k.value" "0.6058"                    "$WORK/manifest.json"
jq_eq "delta rows populated"     ".delta_rows | length"       "1"                          "$WORK/manifest.json"
jq_eq "gate threshold math=3pp"  ".quality_gates.per_task_threshold_pct.gsm8k" "3.0"      "$WORK/manifest.json"
jq_eq "gate triggered"           ".auto_publish_blocked"      "true"                       "$WORK/manifest.json"
jq_eq "override reason recorded" '.gate_override_reason | startswith("CI fixture")' "true" "$WORK/manifest.json"
jq_eq "headline non-empty"       '.headline | length > 0'     "true"                       "$WORK/manifest.json"
jq_eq "render_card_py_sha present" '.tooling.render_card_py_sha256 | length' "64"          "$WORK/manifest.json"

echo "=== REPRODUCE.sh ==="
check "reproduce references model id" "fixture/SampleModel" "$WORK/REPRODUCE.sh"
check "reproduce is executable shebang" "#!/usr/bin/env bash" "$WORK/REPRODUCE.sh"
[ -x "$WORK/REPRODUCE.sh" ] && { echo "  PASS: REPRODUCE.sh is executable"; pass=$((pass+1)); } \
                             || { echo "  FAIL: REPRODUCE.sh not executable"; fail=$((fail+1)); }

echo ""
echo "=== SUMMARY: $pass passed, $fail failed ==="
[ "$fail" -eq 0 ] || exit 1
