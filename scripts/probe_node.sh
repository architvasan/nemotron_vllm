#!/bin/bash
echo "=== HOST ==="; hostname; date
echo "=== GPUs ==="; nvidia-smi -L 2>&1
echo "=== GPU mem ==="; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>&1
echo "=== CPU/RAM ==="; echo "nproc=$(nproc)"; free -g | head -2
echo "=== NVLink topo ==="; nvidia-smi topo -m 2>&1 | head -20
echo "=== internet from compute node? ==="; curl -sS -m 8 -I https://huggingface.co 2>&1 | head -1 || echo "NO INTERNET on compute node"
echo "=== local scratch on node ==="; df -h /tmp /local* /scratch* 2>/dev/null | grep -v Filesystem
echo "=== apptainer present on node ==="; command -v apptainer && apptainer --version
echo "=== DONE ==="
