#!/bin/bash
# Runs INSIDE the apptainer container (one long-lived `apptainer exec` per node,
# launched by ray_vllm_node.sh). Branches on $RANK: rank 0 is the Ray HEAD and
# then runs `vllm serve` in THIS shell (so the head's ray daemons stay alive for
# the life of the server); every other rank joins as a WORKER with `ray start
# --block` (which holds its raylet up until the cluster stops).
#
# Why this shape (vs. an apptainer *instance* + separate execs): starting the ray
# head in one short-lived exec and vllm in another left the GCS unreachable at
# serve time. Keeping ray + vllm in a single session is the pattern vLLM's own
# multi-node launcher uses, and it removes that failure mode.
#
# Ray uses its DEFAULT temp dir (/tmp/ray) on purpose: it is short (no long unix
# socket paths) and node-local. We do NOT set RAY_ADDRESS in the env (that env var
# is for clients finding a cluster and misleads `ray start --head`); the head and
# its readiness probe use the explicit $HEAD_IP:$RAY_PORT instead.
set -x
RANK=${RANK:-0}
RAY_ADDR="$HEAD_IP:$RAY_PORT"

dump_ray_logs() {   # surface GCS/raylet errors to stdout + persist to /grand
  echo "===== ray session logs (${1:-head}) ====="
  local d=/tmp/ray/session_latest/logs
  for f in gcs_server.err gcs_server.out raylet.err raylet.out; do
    [ -f "$d/$f" ] && { echo "--- $f ---"; tail -40 "$d/$f"; }
  done
  mkdir -p "$BASE/logs/ray_${1:-head}_$(hostname)" 2>/dev/null
  cp -r "$d" "$BASE/logs/ray_${1:-head}_$(hostname)/" 2>/dev/null || true
}

if [ "$RANK" != "0" ]; then
  # ---------------- WORKER ----------------
  echo "[rank $RANK] worker $NODE_IP waiting for head $RAY_ADDR"
  for i in $(seq 1 180); do
    (exec 3<>"/dev/tcp/$HEAD_IP/$RAY_PORT") 2>/dev/null && { exec 3>&- 3<&-; break; }
    sleep 2
  done
  echo "[rank $RANK] head reachable; joining and blocking"
  exec ray start --address="$RAY_ADDR" --node-ip-address="$NODE_IP" \
       --num-gpus=4 --block
fi

# ---------------- HEAD ----------------
echo "[rank 0] starting ray head on $RAY_ADDR"
ray start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" --num-gpus=4

# Fail FAST if the head's own GCS isn't reachable locally (don't wait 10 min).
echo "[rank 0] probing local GCS ..."
if ! timeout 60 bash -c "until ray status --address '$RAY_ADDR' >/dev/null 2>&1; do sleep 3; done"; then
  echo "[rank 0] FATAL: head GCS never came up at $RAY_ADDR"
  ray status --address "$RAY_ADDR" || true
  dump_ray_logs head
  exit 1
fi
echo "[rank 0] head GCS is up"

# Wait until every node's GPUs have registered (8 total for 2 nodes).
echo "[rank 0] waiting for $EXPECT_GPUS GPUs to register ..."
if ! RAY_ADDR="$RAY_ADDR" EXPECT_GPUS="$EXPECT_GPUS" python3 - <<'PYEOF'
import ray, os, sys, time
want = int(os.environ["EXPECT_GPUS"])
ray.init(address=os.environ["RAY_ADDR"], log_to_driver=False)
deadline = time.time() + 300
g = 0
while time.time() < deadline:
    g = sum(n.get("Resources", {}).get("GPU", 0) for n in ray.nodes() if n["Alive"])
    if g >= want:
        print(f"GPUS_READY {g}/{want}"); sys.exit(0)
    time.sleep(5)
print(f"TIMEOUT waiting for GPUs: {g}/{want}"); sys.exit(1)
PYEOF
then
  echo "ray cluster never reached $EXPECT_GPUS GPUs"
  ray status --address "$RAY_ADDR" || true
  dump_ray_logs head
  exit 1
fi
ray status --address "$RAY_ADDR"

# ---------------- serve (keeps the head session, hence ray, alive) ----------------
echo "[rank 0] launching vllm serve"
vllm serve "$MODEL" \
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

echo "[rank 0] vllm exited; stopping ray"
ray stop || true
