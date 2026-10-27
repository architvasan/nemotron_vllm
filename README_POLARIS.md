# Nemotron-3.5-Super-VL-120B on ALCF Polaris (vLLM, multi-node)

Polaris port of the JLSE recipe. Serves
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` across **2 Polaris nodes
(8× A100-40 GB)** with vLLM + Ray, reachable from your laptop over an SSH tunnel
as an OpenAI-compatible endpoint.

## Why this differs from the JLSE recipe

| | JLSE | Polaris |
|---|---|---|
| Scheduler | Cobalt (`qsub -t <min> -n <nodes>`) | **PBS Pro** (`qsub -l select=… -q … -A …`) |
| GPU | 4× **H100-80 GB** (320 GB/node) | 4× **A100-40 GB** (160 GB/node) |
| Fit | whole model on **1 node**, TP4 | **doesn't fit 1 node** → **2 nodes**, TP4×PP2 + Ray |
| Container build | podman→tar→compute (8 GB login cgroup) | `apptainer build` on a **compute node** (login cgroup thread cap hangs it) |
| Ray | in base image | **not** in `vllm-openai` image → added via a `:ro` overlay |
| Weights | `/vast/users/avasan/nemotron` | `/grand/FRAME-IDP/avasan/nemotron` (`-l filesystems=home:grand`) |
| Internet | login only | login + **compute** (both via `proxy.alcf.anl.gov:3128`) |

**The memory math:** BF16 weights are ~249 GB; one Polaris node is 160 GB. Split
over 8 GPUs that's ~31 GB of weights/GPU, leaving ~5–6 GB/GPU for KV cache —
hence the conservative `--max-model-len 16384`, `--max-num-seqs 4`, and
`--enforce-eager` on first bring-up. Loosen them once it's stable (see Tuning).

**The topology:** `--tensor-parallel-size 4` keeps the heavy tensor-parallel
all-reduces **inside** a node (NVLink); `--pipeline-parallel-size 2` crosses the
node boundary, which only passes pipeline activations — the cheap direction.
Cross-node collectives run as **NCCL-over-TCP on the Slingshot `hsn0` interface**
(`NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=hsn0`) so the path is guaranteed to
route first; switch to verbs/OFI later for throughput.

## Layout (`/grand/FRAME-IDP/avasan/nemotron/`)
```
model/                       # the 249 GB weights (63 shards + configs)
vllm-0.32.0.sif              # vLLM container (apptainer)
ray-overlay.img              # ext3 overlay that adds ray to the SIF (mounted :ro)
venv/                        # HF-download venv (built from conda/2026-10-01 python)
ENDPOINT.txt                 # head node / port, rewritten each launch
logs/                        # hf_download.log, build.log, DONE markers
<repo>/serve_nemotron_polaris.sh     # the PBS job (select=2)
<repo>/scripts/ray_vllm_node.sh      # per-node Ray head/worker + vllm launcher
<repo>/scripts/0{1,2,3}_*_polaris.sh # download / build / submit
<repo>/scripts/04_test_inference.sh  # endpoint smoke test (shared with JLSE)
```

## One-time environment
```bash
module use /soft/modulefiles
module load spack-pe-base/0.10.1 apptainer/1.4.1   # apptainer 1.4.1
```

## 0. Open a persistent SSH connection (once, with your MFA token)
```bash
# in ~/.ssh/config:  Host polaris → HostName polaris.alcf.anl.gov, User avasan,
#   ControlMaster auto, ControlPath ~/.ssh/cm-%C, ControlPersist 8h
ssh polaris "echo connected on \$(hostname)"   # prompts for the OTP once
```
After that, `ssh polaris <cmd>` reuses the socket with no prompt.

## 1. Download the weights — `scripts/01_download_weights_polaris.sh`
Run on a **login** node. Builds a venv (conda/2026-10-01 python) and downloads
~249 GB into `model/` via Xet, inside a detached `tmux` session (`nemodl`).
```bash
ssh polaris "cd /grand/FRAME-IDP/avasan/nemotron/<repo> && bash scripts/01_download_weights_polaris.sh"
# watch: tail -f /grand/FRAME-IDP/avasan/nemotron/logs/hf_download.log
# done:  ls .../model/model-*-of-*.safetensors | wc -l   → 63
```

## 2. Build the container — `scripts/02_build_container_polaris.sh`
The SIF must be built on a **compute node**, not the login node: on the login
node rootless `apptainer build` / `mksquashfs` hang against the cgroup thread+pids
cap ("Failed to create thread") and rootless `podman pull` crawls at 0 B/s.
`02_build_container_polaris.sh` just `qsub`s the real build job
(`scripts/02b_build_sif_polaris.pbs`, debug queue, ~8 min) which pulls from Docker
Hub through the proxy with `APPTAINER_TMPDIR` on node-local `/local/scratch`.
```bash
ssh polaris "cd .../<repo> && bash scripts/02_build_container_polaris.sh"   # submits 02b
# watch: qstat -xf <jobid> | grep job_state ; done at 'BUILD_DONE_OK'
# done:  ls -lh /grand/FRAME-IDP/avasan/nemotron/vllm-0.32.0.sif
```

### 2b. Add Ray — `scripts/02c_build_ray_overlay.sh`
The `vllm-openai` image ships **without ray**, which multi-node vLLM needs as its
`--distributed-executor-backend`. Bake ray into a small ext3 overlay (mounted
`:ro` at serve time). Run on a compute node (needs the proxy for pip):
```bash
# grab an interactive node, then run the script:
ssh polaris "qsub -I -A FRAME-IDP -q debug -l select=1:system=polaris \
  -l walltime=00:30:00 -l filesystems=home:grand"
#   (inside)  bash .../<repo>/scripts/02c_build_ray_overlay.sh
# done: ls -lh /grand/FRAME-IDP/avasan/nemotron/ray-overlay.img  (verify prints 'OK ray ... vllm ...')
```

## 3. Serve — `serve_nemotron_polaris.sh` (PBS, 2 nodes)
```bash
ssh polaris "cd .../<repo> && bash scripts/03_submit_serve_polaris.sh debug 01:00:00"
# or directly:  qsub serve_nemotron_polaris.sh
```
The job starts a Ray head on node 0 and a worker on node 1, waits for all 8 GPUs
to register, then launches `vllm serve`. Startup ≈ a few minutes (load 249 GB +
init). Ready when the log shows **`Application startup complete`**.
```bash
qstat -x -u avasan                                  # R = running
cat /grand/FRAME-IDP/avasan/nemotron/ENDPOINT.txt   # head node + port
tail -f <jobid>.OU                                  # vLLM log
```

## 4. Test — `scripts/04_test_inference.sh`
```bash
# on the login node, hitting the head compute node directly:
bash scripts/04_test_inference.sh <headnode>:8000
# from your laptop after the tunnel (below):
bash scripts/04_test_inference.sh localhost:8000
```

## Connect from your laptop
The server listens on the **head compute node**; reach it through the login node:
```bash
# read ENDPOINT.txt for the node name, then:
ssh -N -L 8000:<headnode>:8000 polaris
```
Point any OpenAI client at:
- **base_url**: `http://localhost:8000/v1`
- **model**:    `nemotron-3.5-super-vl`
- **api_key**:  any non-empty string (vLLM ignores it), e.g. `EMPTY`

> Reasoning model: give it `max_tokens >= 512`. Thinking text is in
> `choices[0].message.reasoning`; the answer is in `choices[0].message.content`.

## Wire into Hermes / Loam
This endpoint needs **no token** (unlike the managed ALCF Inference Service that
`scripts/alcf_inference_proxy.py` injects a Bearer token for) — just the tunnel.
With the tunnel up, add to the Hermes `config.yaml`:
```yaml
custom_providers:
  - name: Nemotron on Polaris (localhost:8000)
    base_url: http://localhost:8000/v1
    api_key: EMPTY
    model: nemotron-3.5-super-vl
```
and set `model.default` / `model.provider` to it to route the campaign reasoner
at the Polaris-hosted Nemotron.

## Tuning (edit `serve_nemotron_polaris.sh` env knobs)
- **More KV / longer context**: raise `MAX_MODEL_LEN` and/or `GPU_MEM_UTIL`
  (0.90 → 0.93) once you confirm it doesn't OOM.
- **CUDA graphs (faster decode)**: `ENFORCE_EAGER=` (empty) to re-enable graph
  capture — costs a bit of VRAM.
- **More concurrency**: raise `MAX_NUM_SEQS`.
- **Longer-lived endpoint**: submit to `preemptable` with a large walltime.
- **TP8 fallback**: if pipeline-parallel misbehaves on this hybrid Mamba-2 arch,
  set `TP=8 PP=1` (tensor-parallel across both nodes). Heavier cross-node
  traffic, but a single parallel dimension — try it if PP2 fails to start.
