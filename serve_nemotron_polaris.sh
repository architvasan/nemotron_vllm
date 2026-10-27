#!/bin/bash
# PBS job: serve NVIDIA Nemotron-3.5-Super-VL-120B across 2 Polaris nodes
# (8x A100-40GB) via vLLM + Ray, TP4 x PP2. OpenAI-compatible on the HEAD node.
#
# The full BF16 model is ~249 GB and will NOT fit on one Polaris node
# (4x A100-40GB = 160 GB), so we span 2 nodes. Submit with:
#   qsub serve_nemotron_polaris.sh
#   # or: bash scripts/03_submit_serve_polaris.sh debug 2
#
#PBS -N nemotron_vllm
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand
#PBS -q debug
#PBS -A FRAME-IDP
set -x

BASE=/grand/FRAME-IDP/avasan/nemotron
export BASE
export SIF=$BASE/vllm-0.32.0.sif
export MODEL=$BASE/model
export PORT=8000
export RAY_PORT=6379

# Topology (override from the submit wrapper if needed).
export TP=${TP:-4}                 # tensor-parallel: WITHIN a node (NVLink)
export PP=${PP:-2}                 # pipeline-parallel: ACROSS nodes
# First-bring-up safety knobs (40 GB cards, ~31 GB/GPU of weights):
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
export ENFORCE_EAGER=${ENFORCE_EAGER:-1}   # skip CUDA-graph capture to save VRAM; unset for speed

module use /soft/modulefiles
module load spack-pe-base/0.10.1
module load apptainer/1.4.1

cd "${PBS_O_WORKDIR:-$BASE}"
mkdir -p $BASE/xdg $BASE/cache $BASE/triton $BASE/inductor $BASE/vllm_cache $BASE/tmp $BASE/logs

# ---- node list: PBS_NODEFILE has one line per node on Polaris ----
mapfile -t NODES < "$PBS_NODEFILE"
export NNODES=${#NODES[@]}
# This script runs on the first node, which is the Ray head. Use its hsn0 IP.
export HEAD_IP=$(ip -4 -o addr show hsn0 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
[ -z "$HEAD_IP" ] && export HEAD_IP=$(hostname -I | awk '{print $1}')
echo "NNODES=$NNODES HEAD=$(hostname) HEAD_IP=$HEAD_IP TP=$TP PP=$PP"

NODE_SCRIPT="${PBS_O_WORKDIR:-$BASE}/scripts/ray_vllm_node.sh"
[ -x "$NODE_SCRIPT" ] || NODE_SCRIPT="$BASE/scripts/ray_vllm_node.sh"
chmod +x "$NODE_SCRIPT" 2>/dev/null || true

# One rank per node; the per-node script decides head vs worker from PMI_RANK.
# --cpu-bind none: Ray/vLLM place their own worker threads; we don't want PALS
# pinning the single rank to one core.
mpiexec -n "$NNODES" --ppn 1 --cpu-bind none \
  bash "$NODE_SCRIPT"
