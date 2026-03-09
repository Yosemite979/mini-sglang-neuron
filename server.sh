#!/usr/bin/env bash
set -euo pipefail

if ! command -v ninja >/dev/null 2>&1; then
  echo "ERROR: 'ninja' is required but was not found in PATH." >&2
  echo "Install it with: apt-get install ninja-build  OR  conda install -c conda-forge ninja" >&2
  exit 1
fi

if ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
  echo "ERROR: A C++ compiler is required (g++ or clang++) but none was found in PATH." >&2
  exit 1
fi

export TP_SIZE=2
export NEURON_RT_NUM_CORES="${TP_SIZE}"
python -m minisgl \
  --model-path /root/data/Qwen/Qwen3-0.6B \
  --dtype bfloat16 \
  --tp-size "$TP_SIZE" \
  --max-running-requests 4 \
  --max-seq-len-override 1024 \
  --num-pages 10192 \
  --port 1919 \
  --cache-type radix

# After starting the server, you can test it with:
#   python3 benchmark/online/simple_call.py --prompt "hello" --max-tokens 500 --temperature 0.6 --top-k -1
