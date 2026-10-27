#!/bin/bash
# Generalized serve: auto-detect GPU count, pick a valid power-of-2 tensor-parallel size (<=8).
# Works on H100, B200, etc. Submit e.g.:
#   qsub -q gpu_b200 -t 480 -n 1 /vast/users/avasan/nemotron/serve_nemotron_auto.sh
set -x
BASE=/vast/users/avasan/nemotron
SIF=$BASE/vllm-0.32.0.sif
MODEL=$BASE/model
PORT=8000
NODE=$(hostname)

# how many GPUs on this node?
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
# largest power of two <= NGPU and <= 8
TP=1; for c in 2 4 8; do [ "$NGPU" -ge "$c" ] && TP=$c; done
echo "Detected NGPU=$NGPU -> tensor-parallel-size=$TP on $NODE"

mkdir -p $BASE/xdg $BASE/cache $BASE/triton $BASE/inductor $BASE/vllm_cache $BASE/tmp

cat > $BASE/ENDPOINT.txt <<EOT
NODE=$NODE
PORT=$PORT
NGPU=$NGPU
TP=$TP
model_name=nemotron-3.5-super-vl
base_url=http://$NODE:$PORT/v1
# From laptop:  ssh -N -L 8000:$NODE:$PORT <user>@jlselogin6.ftm.alcf.anl.gov
# Then point client at  http://localhost:8000/v1
started=$(date)
EOT
cat $BASE/ENDPOINT.txt

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
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --limit-mm-per-prompt '{"image": 1, "video": 1}' \
    --allowed-local-media-path / \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3
