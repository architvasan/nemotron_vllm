#!/bin/bash
# 02c (Polaris) - Add Ray to the vLLM image via a read-only apptainer overlay.
#
# WHY: vllm/vllm-openai:v0.32.0 ships WITHOUT ray (ray is an optional extra, not
# in the image's core requirements), but multi-node vLLM uses ray as its
# --distributed-executor-backend. Rather than rebuild the whole image, we bake
# ray into a small ext3 overlay file that sits next to the SIF. At serve time the
# overlay is mounted READ-ONLY on every node: a single ext3-in-a-file overlay
# cannot be writable from two nodes at once, and ray only READS its installed
# package files at runtime (its scratch/logs go to node-local /tmp), so :ro is
# both required and sufficient.
#
# Run this on a COMPUTE node (it pip-installs, so it needs outbound via the proxy)
# -- e.g. grab an interactive node:  qsub -I -A FRAME-IDP -q debug \
#   -l select=1:system=polaris -l walltime=00:30:00 -l filesystems=home:grand
# then run this script. The pip step is slow (minutes) only because the
# FUSE-backed ext3 overlay has to create ray's thousands of small files.
set -ex
module use /soft/modulefiles
module load spack-pe-base/0.10.1
module load apptainer/1.4.1

BASE=${BASE:-/grand/FRAME-IDP/avasan/nemotron}
SIF=${SIF:-$BASE/vllm-0.32.0.sif}
OV=${OVERLAY:-$BASE/ray-overlay.img}
export HTTP_PROXY=${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}
export HTTPS_PROXY=${HTTPS_PROXY:-http://proxy.alcf.anl.gov:3128}
export http_proxy=${http_proxy:-$HTTP_PROXY}
export https_proxy=${https_proxy:-$HTTPS_PROXY}

# 1. create a 4 GB ext3 overlay file (rootless; no fakeroot needed)
rm -f "$OV"
apptainer overlay create --size 4096 "$OV"

# 2. install ray INTO the overlay (writable mount)
apptainer exec --overlay "$OV" --bind /grand "$SIF" \
  python3 -m pip install --no-cache-dir ray
echo "PIP_EXIT=$?"

# 3. verify ray + vllm import with the overlay mounted READ-ONLY (serve-time mode)
apptainer exec --overlay "${OV}:ro" --bind /grand "$SIF" \
  python3 -c 'import ray, vllm; print("OK ray", ray.__version__, "vllm", vllm.__version__)'
echo "VERIFY_EXIT=$?"
echo "Overlay ready: $OV  (mount it :ro at serve time)"
