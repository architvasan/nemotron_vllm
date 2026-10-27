# Nemotron-3.5-Super-VL-120B on JLSE (vLLM)

> **Running on ALCF Polaris instead?** See **[README_POLARIS.md](README_POLARIS.md)** —
> the model doesn't fit one Polaris node (4× A100-40 GB), so that recipe serves it
> across **2 nodes** with Ray + TP4×PP2 under PBS.

End-to-end recipe to download, serve, and query
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` on JLSE GPUs via vLLM,
reachable from your laptop over an SSH tunnel as an OpenAI-compatible endpoint.

## Overview

- **Model**: hybrid Mamba-2 + Transformer + MoE, vision+text, 120B total / 12B active, BF16 (~249 GB, 63 shards).
- **Engine**: vLLM 0.32.0 in an apptainer image (`vllm-0.32.0.sif`).
- **Hardware**: validated on `hopper00` = 4× H100 80 GB (tensor-parallel 4). Also queue-able on B200 (`gpu_b200`, auto-TP).
- **API**: OpenAI-compatible, served model name `nemotron-3.5-super-vl`.
- **Everything lives on** `/vast/users/avasan/nemotron/` (because `/home` has a ~100 GB quota that's nearly full; `/vast` is mounted on the compute nodes over RDMA).

### JLSE-specific gotchas handled below
- The login-node user cgroup is capped at **8 GB RAM**, so `apptainer pull` / `mksquashfs` get OOM-killed there → build the SIF in a compute-node job instead.
- Rootless **podman has no subuid range** here → pull with `overlay` + `ignore_chown_errors`.
- The **gpu_h100 queue max walltime is 6 h** (gpu_b200 is 8 h); the endpoint lives only while the job runs, so relaunch when it expires.

### Layout
```
/vast/users/avasan/nemotron/
├── model/                     # the 249 GB weights (63 shards + configs)
├── vllm-0.32.0.sif            # vLLM container (apptainer)
├── serve_nemotron.sh          # serve job: fixed TP4 (gpu_h100)
├── serve_nemotron_auto.sh     # serve job: auto-detect GPUs -> TP (any node, e.g. gpu_b200)
├── ENDPOINT.txt               # node/port, rewritten on each launch
├── README.md                  # this file
├── README_CONNECT.md          # connection cheat-sheet
├── logs/                      # download/build logs
└── scripts/
    ├── 01_download_weights.sh
    ├── 02_build_container.sh
    ├── 03_submit_serve.sh
    └── 04_test_inference.sh
```

---

## 1. Download the weights — `scripts/01_download_weights.sh`
Run on the **login node** (has internet). Goes to `/vast`.

```bash
#!/bin/bash
set -euo pipefail
BASE=/vast/users/avasan/nemotron
REPO=nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16
mkdir -p "$BASE"/{cache,tmp,model,logs}

# one-time venv (system python is 3.6, too old)
if [ ! -x "$BASE/venv/bin/hf" ]; then
  /usr/bin/python3.11 -m venv "$BASE/venv"
  "$BASE/venv/bin/pip" install --quiet --upgrade pip
  "$BASE/venv/bin/pip" install --quiet "huggingface_hub[cli]" hf_xet   # Xet = fast transfer
fi

export HF_HOME="$BASE/cache" TMPDIR="$BASE/tmp"
"$BASE/venv/bin/hf" download "$REPO" --local-dir "$BASE/model" 2>&1 | tee "$BASE/logs/hf_download.log"
echo "shards: $(ls "$BASE"/model/model-*-of-*.safetensors | wc -l)/63"; du -sh "$BASE/model"
```

---

## 2. Build the container — `scripts/02_build_container.sh`
Two stages: podman pulls + tars on the login node; apptainer builds the SIF in a short compute-node job.

```bash
#!/bin/bash
set -euo pipefail
BASE=/vast/users/avasan/nemotron
IMAGE=docker://docker.io/vllm/vllm-openai:v0.32.0
OCITAR=$BASE/vllm-0.32.0.oci.tar

# Stage A (LOGIN): pull with podman into LOCAL /tmp, export an OCI tarball to /vast
mkdir -p /tmp/podman-store /tmp/podman-run /tmp/podman-tmp; export TMPDIR=/tmp/podman-tmp
podman --root /tmp/podman-store --runroot /tmp/podman-run --storage-driver overlay \
  pull --storage-opt overlay.ignore_chown_errors=true "$IMAGE"
podman --root /tmp/podman-store --runroot /tmp/podman-run --storage-driver overlay \
  save --format oci-archive -o "$OCITAR" docker.io/vllm/vllm-openai:v0.32.0

# Stage B (COMPUTE job): tarball -> SIF where there's no 8 GB cap
cat > /home/avasan/nemo_build_sif.sh <<'EOF'
#!/bin/bash
set -ex
BASE=/vast/users/avasan/nemotron
export APPTAINER_TMPDIR=/tmp/apptbuild_$$ APPTAINER_CACHEDIR=/tmp/apptcache_$$
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
apptainer build "$BASE/vllm-0.32.0.sif" "oci-archive:$BASE/vllm-0.32.0.oci.tar"
ls -lh "$BASE/vllm-0.32.0.sif"; echo BUILD_DONE_OK
EOF
chmod +x /home/avasan/nemo_build_sif.sh
cd /home/avasan && qsub -q gpu_h100 -t 25 -n 1 /home/avasan/nemo_build_sif.sh
# when done: rm -f "$OCITAR"
```

---

## 3. Serve on the GPUs — `serve_nemotron.sh`
Submit with:
```bash
qsub -q gpu_h100 -t 360 -n 1 /vast/users/avasan/nemotron/serve_nemotron.sh
# or: bash /vast/users/avasan/nemotron/scripts/03_submit_serve.sh gpu_h100
# B200 (8h, auto-TP): bash .../scripts/03_submit_serve.sh gpu_b200
```

The exact job that produced the working endpoint:
```bash
#!/bin/bash
set -x
BASE=/vast/users/avasan/nemotron
SIF=$BASE/vllm-0.32.0.sif; MODEL=$BASE/model; PORT=8000; NODE=$(hostname)
mkdir -p $BASE/xdg $BASE/cache $BASE/triton $BASE/inductor $BASE/vllm_cache $BASE/tmp

cat > $BASE/ENDPOINT.txt <<EOT
NODE=$NODE
PORT=$PORT
model_name=nemotron-3.5-super-vl
base_url=http://$NODE:$PORT/v1
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
```

`serve_nemotron_auto.sh` is identical except it sets
`--tensor-parallel-size` to the largest power-of-two ≤ the node's GPU count (≤ 8),
so it works on B200 / other nodes without editing.

Startup takes ~4 min (load 249 GB + CUDA-graph capture). It's ready when the log shows
`Application startup complete`. Track:
```bash
qstat -u avasan                      # wait for State=running, note Location (node)
cat /vast/users/avasan/nemotron/ENDPOINT.txt
tail -f /home/avasan/<jobid>.output
```

---

## 4. Test inference — `scripts/04_test_inference.sh`
Pass `host:port`. On the login node hit the node directly; on your laptop use `localhost` after the tunnel.

```bash
#!/bin/bash
set -euo pipefail
HP=${1:-localhost:8000}; URL="http://$HP"; MODEL=nemotron-3.5-super-vl

echo "=== 1) GET /v1/models ==="
curl -sS -m 15 "$URL/v1/models" | python3 -m json.tool

echo "=== 2) chat (reasoning model: big max_tokens; split reasoning vs content) ==="
curl -sS -m 120 "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*24? Reply with just the number.\"}],\"max_tokens\":1024,\"temperature\":0}" \
| python3 -c "import sys,json;c=json.load(sys.stdin)['choices'][0];print('finish:',c['finish_reason']);print('reasoning:',c['message'].get('reasoning'));print('content:',c['message'].get('content'))"

echo "=== 3) streaming (first chunks) ==="
curl -sS -N -m 60 "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"stream\":true,\"max_tokens\":256,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in 3 languages.\"}]}" | head -8
```

> **Reasoning model note:** thinking text is in `choices[0].message.reasoning`, the
> final answer in `choices[0].message.content`. Use `max_tokens >= 512` or it spends
> the whole budget thinking and `content` comes back empty.

---

## Connect from your laptop (OpenAI-compatible client / "hermes")
The server listens on the **compute node**; reach it through the login node:
```bash
ssh -N -L 8000:hopper00:8000 avasan@jlselogin6.ftm.alcf.anl.gov
# (if the node isn't hopper00, check ENDPOINT.txt and retarget the forward)
```
Then point the client at:
- **base_url**: `http://localhost:8000/v1`
- **model**:    `nemotron-3.5-super-vl`
- **api_key**:  any non-empty string (ignored by vLLM), e.g. `EMPTY`

Vision (it's a VL model): send `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`
in the message `content` list. `--allowed-local-media-path /` also allows `file://` paths on JLSE.

---

## Quick reference — full end to end
```bash
bash /vast/users/avasan/nemotron/scripts/01_download_weights.sh      # once (~249 GB)
bash /vast/users/avasan/nemotron/scripts/02_build_container.sh       # once (builds the SIF)
qsub -q gpu_h100 -t 360 -n 1 /vast/users/avasan/nemotron/serve_nemotron.sh
# wait for running, then from the login node:
bash /vast/users/avasan/nemotron/scripts/04_test_inference.sh hopper00:8000
# from your laptop:
ssh -N -L 8000:hopper00:8000 avasan@jlselogin6.ftm.alcf.anl.gov
bash /vast/users/avasan/nemotron/scripts/04_test_inference.sh localhost:8000
```
The weights (#1) and container (#2) are already done on disk — going forward you only need #3 and #4.

## Tuning (edit the serve script)
- Longer context: raise `--max-model-len` (KV room ~1.97M tokens total at 32k on 4× H100).
- More concurrency: raise `--max-num-seqs`.
- Faster decode: add MTP speculative decoding —
  `--speculative_config.method mtp --speculative_config.num_speculative_tokens 3`.
