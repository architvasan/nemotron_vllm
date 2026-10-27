#!/bin/bash
# 02 (Polaris) - Build the vLLM apptainer image (SIF) on a COMPUTE node.
#
# IMPORTANT: the SIF must be built on a COMPUTE node, NOT a login node.
# We tried the login node first (direct `apptainer build docker://...`) and it
# hangs: rootless layer extraction / mksquashfs run into the login-node cgroup
# thread+pids cap ("mksquashfs: Failed to create thread"), and rootless podman
# pull crawls on the overlay chown-skip with 0 bytes of I/O. Compute nodes have
# no such cap, have fast local NVMe (/local/scratch) for APPTAINER_TMPDIR, and
# can still reach Docker Hub outbound through the ALCF proxy.
#
# So this script just SUBMITS the real build job (scripts/02b_build_sif_polaris.pbs)
# to the debug queue and tells you how to watch it. ~8 min wall once it starts.
#
# After the SIF exists, add Ray to it with scripts/02c_build_ray_overlay.sh
# (the vllm-openai image ships without ray, which multi-node vLLM needs).
set -euo pipefail

BASE=/grand/FRAME-IDP/avasan/nemotron
SIF=$BASE/vllm-0.32.0.sif
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SIF" ]; then
  echo "SIF already exists: $SIF"
  ls -lh "$SIF"
  echo "Nothing to build. (Remove it first if you want a fresh build.)"
  exit 0
fi

JOBID=$(qsub "$HERE/02b_build_sif_polaris.pbs")
echo "Submitted SIF build job: $JOBID"
echo "Watch it:"
echo "  qstat -xf $JOBID | grep job_state"
echo "  tail -f \$(ls -t $PWD/nemo_build_sif.o* 2>/dev/null | head -1)"
echo "Done when the job log ends with 'BUILD_DONE_OK' and the SIF exists:"
echo "  ls -lh $SIF"
echo
echo "Next: bash $HERE/02c_build_ray_overlay.sh   # bake Ray into a :ro overlay"
