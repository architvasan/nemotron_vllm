#!/bin/bash
# Per-node launcher, run once per node by mpiexec (`--ppn 1`) from
# serve_nemotron_polaris.sh. It sets up this node's env and does ONE long-lived
# `apptainer exec` of ray_vllm_incontainer.sh, which (by $RANK) is either the Ray
# HEAD + `vllm serve`, or a Ray WORKER that blocks. Keeping ray and vllm in a
# single container session is what keeps the head's GCS alive (an earlier
# apptainer-*instance* + separate-exec design left the GCS unreachable at serve
# time).
#
# Topology TP=4 x PP=2: tensor-parallel all-reduces stay INSIDE a node (NVLink);
# only pipeline activations cross nodes. Cross-node collectives run as NCCL-over-
# TCP on the Slingshot hsn0 interface (NCCL_IB_DISABLE=1, NCCL_SOCKET_IFNAME=hsn0).
set -x

RANK=${PMI_RANK:-${PALS_RANKID:-0}}

: "${BASE:?}"; : "${SIF:?}"; : "${MODEL:?}"; : "${HEAD_IP:?}"
PORT=${PORT:-8000}
RAY_PORT=${RAY_PORT:-6379}
TP=${TP:-4}
PP=${PP:-2}
NNODES=${NNODES:-2}
EXPECT_GPUS=$(( NNODES * 4 ))

# This node's own IP on the Slingshot fabric (fall back to the first address).
NODE_IP=$(ip -4 -o addr show hsn0 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
[ -z "$NODE_IP" ] && NODE_IP=$(hostname -I | awk '{print $1}')

# Ray is baked into a read-only overlay (the vllm-openai image ships without it).
# RO is required: one ext3-in-a-file overlay can't be writable from two nodes at
# once; ray only READS its package files, runtime writes go to node-local /tmp.
OVERLAY=${OVERLAY:-$BASE/ray-overlay.img}

# The in-container script lives beside this one on /grand (bind-mounted, so its
# absolute path is valid inside the container too).
INCONTAINER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ray_vllm_incontainer.sh"

# Head publishes the endpoint for the login-node / laptop tunnel.
if [ "$RANK" = "0" ]; then
  cat > "$BASE/ENDPOINT.txt" <<EOT
NODE=$(hostname)
HEAD_IP=$HEAD_IP
PORT=$PORT
NNODES=$NNODES
TP=$TP
PP=$PP
model_name=nemotron-3.5-super-vl
base_url=http://$(hostname):$PORT/v1
# From laptop:  ssh -N -L 8000:$(hostname):$PORT polaris
# Then point the OpenAI client at  http://localhost:8000/v1
started=$(date)
EOT
  cat "$BASE/ENDPOINT.txt"
fi

exec apptainer exec --nv \
  --overlay "$OVERLAY":ro \
  --bind /grand \
  --env RANK="$RANK" \
  --env HEAD_IP="$HEAD_IP" \
  --env NODE_IP="$NODE_IP" \
  --env RAY_PORT="$RAY_PORT" \
  --env PORT="$PORT" \
  --env TP="$TP" \
  --env PP="$PP" \
  --env EXPECT_GPUS="$EXPECT_GPUS" \
  --env MODEL="$MODEL" \
  --env GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}" \
  --env MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}" \
  --env MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}" \
  --env ENFORCE_EAGER="${ENFORCE_EAGER:-}" \
  --env BASE="$BASE" \
  --env HF_HOME="$BASE/cache" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env XDG_CACHE_HOME="$BASE/xdg" \
  --env VLLM_CACHE_ROOT="$BASE/vllm_cache" \
  --env TRITON_CACHE_DIR="$BASE/triton" \
  --env TORCHINDUCTOR_CACHE_DIR="$BASE/inductor" \
  --env TMPDIR="$BASE/tmp" \
  --env OMP_NUM_THREADS=16 \
  --env VLLM_HOST_IP="$NODE_IP" \
  --env NCCL_IB_DISABLE=1 \
  --env NCCL_SOCKET_IFNAME=hsn0 \
  --env GLOO_SOCKET_IFNAME=hsn0 \
  --env NCCL_DEBUG=WARN \
  "$SIF" bash "$INCONTAINER"
