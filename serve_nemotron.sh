#!/bin/bash
# Cobalt job: serve NVIDIA Nemotron-3.5-Super-VL-120B on 4x H100 (hopper00) via vLLM
# Submit with:  qsub -q gpu_h100 -t 360 -n 1 /vast/users/avasan/nemotron/serve_nemotron.sh
set -x
BASE=/vast/users/avasan/nemotron
SIF=$BASE/vllm-0.32.0.sif
MODEL=$BASE/model
PORT=8000
NODE=$(hostname)

# writable cache/tmp dirs on /vast (container rootfs is read-only)
mkdir -p $BASE/xdg $BASE/cache $BASE/triton $BASE/inductor $BASE/vllm_cache $BASE/tmp

# publish connection info so it can be read from the login node
cat > $BASE/ENDPOINT.txt <<EOT
NODE=$NODE
PORT=$PORT
model_name=nemotron-3.5-super-vl
base_url=http://$NODE:$PORT/v1
# From laptop:  ssh -N -L 8000:$NODE:$PORT <user>@<jlse-login-host>
# Then point client at  http://localhost:8000/v1
started=$(date)
EOT
cat $BASE/ENDPOINT.txt

export APPTAINERENV_HF_HOME=$BASE/cache

apptainer exec --nv \
  --bind /vast \
  --env HF_HOME=$BASE/cache \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env XDG_CACHE_HOME=$BASE/xdg \
  --env VLLM_CACHE_ROOT=$BASE/vllm_cache \
  --env TRITON_CACHE_DIR=$BASE/triton \
  --env TORCHINDUCTOR_CACHE_DIR=$BASE/inductor \
  --env TMPDIR=$BASE/tmp \
  --env OMP_NUM_THREADS=16 \
  "$SIF" \
  vllm serve "$MODEL" \
    --served-model-name nemotron-3.5-super-vl \
    --host 0.0.0.0 --port $PORT \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --limit-mm-per-prompt '{"image": 1, "video": 1}' \
    --allowed-local-media-path / \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3
