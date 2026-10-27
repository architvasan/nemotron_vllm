#!/bin/bash
# Per-node launcher, run once per node by mpiexec (`--ppn 1`) from
# serve_nemotron_polaris.sh. Rank 0 (PMI_RANK=0) is the Ray HEAD and also runs
# `vllm serve`; every other rank is a Ray WORKER that joins the head and blocks
# for the life of the job.
#
# Design notes for multi-node vLLM in apptainer on Polaris (Slingshot):
#   * We use an apptainer INSTANCE per node so the Ray daemons started in the
#     first `exec` survive for the second `exec` (vllm) on the head node.
#   * Topology is TP=4 x PP=2: tensor-parallel all-reduces stay INSIDE a node
#     (NVLink); only pipeline activations cross the node boundary. That is the
#     cheap direction, which matters because cross-node collectives here run as
#     NCCL-over-TCP on the Slingshot IP interface (hsn0), not verbs.
#   * NCCL is pinned to hsn0 sockets (NCCL_IB_DISABLE=1, NCCL_SOCKET_IFNAME=hsn0)
#     so the cross-node path is guaranteed to route. Flip to verbs/OFI later for
#     throughput once the plain-socket version is proven to work.
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

RAY_TMP=/tmp/ray_${USER}_${PBS_JOBID:-0}
INST=nemo

# One apptainer instance on this node, holding the Ray daemons.
apptainer instance stop "$INST" 2>/dev/null || true
apptainer instance start --nv \
  --bind /grand \
  --env HF_HOME=$BASE/cache \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env XDG_CACHE_HOME=$BASE/xdg \
  --env VLLM_CACHE_ROOT=$BASE/vllm_cache \
  --env TRITON_CACHE_DIR=$BASE/triton \
  --env TORCHINDUCTOR_CACHE_DIR=$BASE/inductor \
  --env TMPDIR=$BASE/tmp \
  --env OMP_NUM_THREADS=16 \
  --env VLLM_HOST_IP=$NODE_IP \
  --env RAY_ADDRESS=$HEAD_IP:$RAY_PORT \
  --env NCCL_IB_DISABLE=1 \
  --env NCCL_SOCKET_IFNAME=hsn0 \
  --env GLOO_SOCKET_IFNAME=hsn0 \
  --env NCCL_DEBUG=WARN \
  "$SIF" "$INST"

EXEC="apptainer exec instance://$INST"

if [ "$RANK" != "0" ]; then
  # ---- WORKER: join the head's Ray cluster and block ----
  echo "[rank $RANK] worker $NODE_IP joining ray head $HEAD_IP:$RAY_PORT"
  $EXEC ray start --address="$HEAD_IP:$RAY_PORT" \
    --node-ip-address="$NODE_IP" --num-gpus=4 --temp-dir="$RAY_TMP" --block
  apptainer instance stop "$INST" 2>/dev/null || true
  exit 0
fi

# ---- HEAD (rank 0): start ray head, wait for all GPUs, run vllm serve ----
echo "[rank 0] head $NODE_IP (HEAD_IP=$HEAD_IP) starting ray"
$EXEC ray start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" \
  --num-gpus=4 --temp-dir="$RAY_TMP"

# publish the endpoint for the login-node / laptop tunnel
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

# wait until every worker has joined (EXPECT_GPUS visible to Ray)
echo "[rank 0] waiting for $EXPECT_GPUS GPUs to register with Ray ..."
$EXEC python - "$EXPECT_GPUS" <<'PYEOF'
import ray, sys, time
want = int(sys.argv[1])
ray.init(address="auto", log_to_driver=False)
deadline = time.time() + 600
g = 0
while time.time() < deadline:
    g = sum(n.get("Resources", {}).get("GPU", 0) for n in ray.nodes() if n["Alive"])
    if g >= want:
        print(f"GPUS_READY {g}/{want}"); sys.exit(0)
    time.sleep(5)
print(f"TIMEOUT waiting for GPUs: {g}/{want}"); sys.exit(1)
PYEOF
if [ $? -ne 0 ]; then echo "ray cluster never reached $EXPECT_GPUS GPUs"; $EXEC ray status; exit 1; fi
$EXEC ray status

# ---- serve ----
$EXEC vllm serve "$MODEL" \
  --served-model-name nemotron-3.5-super-vl \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --pipeline-parallel-size "$PP" \
  --distributed-executor-backend ray \
  --trust-remote-code \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --limit-mm-per-prompt '{"image": 1, "video": 1}' \
  --allowed-local-media-path / \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  ${ENFORCE_EAGER:+--enforce-eager}

# vllm exited -> tear the cluster down so the worker ranks unblock and the job ends
$EXEC ray stop || true
apptainer instance stop "$INST" 2>/dev/null || true
