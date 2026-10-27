#!/bin/bash
# 01 - Download Nemotron-3.5-Super-VL-120B weights (~249 GB) to /vast.
# Run on the LOGIN node (jlselogin6) - it has internet; compute nodes may not.
# NOTE: /home has a 100 GB quota (mostly full), so everything lives on /vast.
set -euo pipefail

BASE=/vast/users/avasan/nemotron
REPO=nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16

mkdir -p "$BASE"/{cache,tmp,model,logs}

# One-time: a modern Python venv on /vast (system python3 is 3.6, too old).
if [ ! -x "$BASE/venv/bin/hf" ]; then
  /usr/bin/python3.11 -m venv "$BASE/venv"
  "$BASE/venv/bin/pip" install --quiet --upgrade pip
  # hub 2.x dropped the [hf_transfer] extra in favor of Xet; install hf_xet for speed.
  "$BASE/venv/bin/pip" install --quiet "huggingface_hub[cli]" hf_xet
fi

export HF_HOME="$BASE/cache"
export TMPDIR="$BASE/tmp"

# Downloads straight into $BASE/model (Xet-accelerated). ~249 GB, 63 safetensors shards.
"$BASE/venv/bin/hf" download "$REPO" --local-dir "$BASE/model" 2>&1 | tee "$BASE/logs/hf_download.log"

echo "shards: $(ls "$BASE"/model/model-*-of-*.safetensors 2>/dev/null | wc -l)/63"
du -sh "$BASE/model"
