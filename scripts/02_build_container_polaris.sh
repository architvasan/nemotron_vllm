#!/bin/bash
# 02 (Polaris) - Build the vLLM apptainer image (SIF) on a LOGIN node.
#
# Unlike JLSE (where the login-node cgroup is capped at 8 GB RAM and rootless
# podman has no subuid range, forcing a two-stage podman->tarball->compute-build
# dance), Polaris login nodes can build the SIF directly from the Docker image:
#   module use /soft/modulefiles
#   module load spack-pe-base/0.10.1 apptainer/1.4.1
#   apptainer build <sif> docker://...
# apptainer pulls through the ALCF proxy, so HTTPS_PROXY must be set.
#
# The build's scratch (APPTAINER_TMPDIR/CACHEDIR) goes on /grand because it needs
# tens of GB to unpack layers + assemble the squashfs; the 9.2 GB image expands
# well beyond the login node's local /tmp.
set -euo pipefail

BASE=/grand/FRAME-IDP/avasan/nemotron
IMAGE=docker://docker.io/vllm/vllm-openai:v0.32.0
SIF=$BASE/vllm-0.32.0.sif

module use /soft/modulefiles
module load spack-pe-base/0.10.1
module load apptainer/1.4.1

mkdir -p "$BASE/apptainer_tmp" "$BASE/apptainer_cache" "$BASE/logs"
export APPTAINER_TMPDIR="$BASE/apptainer_tmp"
export APPTAINER_CACHEDIR="$BASE/apptainer_cache"
export HTTPS_PROXY=${HTTPS_PROXY:-http://proxy.alcf.anl.gov:3128}
export HTTP_PROXY=${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}
export https_proxy=${https_proxy:-http://proxy.alcf.anl.gov:3128}
export http_proxy=${http_proxy:-http://proxy.alcf.anl.gov:3128}

# Detached tmux: the build takes ~10-20 min (pull + squashfs) -- keep it alive.
rm -f "$BASE/logs/BUILD_DONE"
tmux kill-session -t nemobuild 2>/dev/null || true
tmux new-session -d -s nemobuild \
  "module use /soft/modulefiles; module load spack-pe-base/0.10.1; module load apptainer/1.4.1; \
   export APPTAINER_TMPDIR=$BASE/apptainer_tmp APPTAINER_CACHEDIR=$BASE/apptainer_cache \
     HTTPS_PROXY=$HTTPS_PROXY HTTP_PROXY=$HTTP_PROXY https_proxy=$https_proxy http_proxy=$http_proxy; \
   apptainer build $SIF $IMAGE > $BASE/logs/build.log 2>&1; \
   echo EXIT=\$? >> $BASE/logs/build.log; touch $BASE/logs/BUILD_DONE"

echo "SIF build started in tmux session 'nemobuild'. Watch it with:"
echo "  tail -f $BASE/logs/build.log"
echo "Done when $BASE/logs/BUILD_DONE appears and $SIF exists:"
echo "  ls -lh $SIF"
echo "You can then delete the build scratch: rm -rf $BASE/apptainer_tmp $BASE/apptainer_cache"
