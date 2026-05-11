#!/bin/bash
set -euo pipefail

# Quick syntax / build pre-check fails fast on broken code.
# Use swift build directly here (mise build is identical, but avoids mise overhead).

PROMPT="${BENCH_PROMPT:-The capital of Ireland is}"
MAX_NEW="${BENCH_MAX_NEW:-64}"
MAX_SEQ="${BENCH_MAX_SEQ:-512}"

# Build (verbose only on failure).
start_wall=$(python3 -c 'import time;print(time.monotonic())')
mise gemma-cli -- --prompt "$PROMPT" --max-new "$MAX_NEW" --max-seq-len "$MAX_SEQ" \
    >bench_stdout.txt 2>bench_stderr.txt
end_wall=$(python3 -c 'import time;print(time.monotonic())')

wall_s=$(python3 -c "print(${end_wall} - ${start_wall})")

prompt_line=$(grep -E '^Prompt:' bench_stderr.txt | tail -1 || true)
gen_line=$(grep -E '^Generation:' bench_stderr.txt | tail -1 || true)

if [ -z "$gen_line" ]; then
    echo "BENCH FAILED — no Generation line. stderr:" >&2
    cat bench_stderr.txt >&2
    exit 1
fi

# "Generation: N tokens, R tok/s"
prompt_tps=$(echo "$prompt_line" | awk -F',' '{print $2}' | awk '{print $1}')
decode_tps=$(echo "$gen_line"    | awk -F',' '{print $2}' | awk '{print $1}')

# Save the actual generated text for sanity check.
cp bench_stdout.txt bench_output.txt 2>/dev/null || true

echo "--- generation ---"
cat bench_output.txt
echo "------------------"
echo "$prompt_line"
echo "$gen_line"

echo "METRIC decode_tps=${decode_tps}"
echo "METRIC prompt_tps=${prompt_tps:-0}"
echo "METRIC wall_s=${wall_s}"
