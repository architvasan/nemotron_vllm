#!/bin/bash
# 03 (Polaris) - Submit the multi-node vLLM serve job via PBS.
# Run on a Polaris login node from the repo dir (so scripts/ray_vllm_node.sh is
# found via $PBS_O_WORKDIR).
#
# Usage: bash scripts/03_submit_serve_polaris.sh [queue] [walltime]
#   queue:    debug (default, 1-2 nodes, <=1h) | debug-scaling | preemptable (<=72h)
#   walltime: HH:MM:SS (default 01:00:00; must fit the queue's limit)
#
# The endpoint lives only while the job runs. For a long-lived endpoint use
# `preemptable` with a big walltime (it can be bumped by `demand` jobs, but for a
# test reasoner that is usually fine).
set -euo pipefail

QUEUE=${1:-debug}
WALLTIME=${2:-01:00:00}
ACCOUNT=${ACCOUNT:-FRAME-IDP}
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # repo root
SERVE="$HERE/serve_nemotron_polaris.sh"

echo "Submitting $SERVE  (queue=$QUEUE walltime=$WALLTIME account=$ACCOUNT)"
JOBID=$(qsub -q "$QUEUE" -l walltime="$WALLTIME" -A "$ACCOUNT" "$SERVE")
echo "submitted: $JOBID"
echo
echo "Track it:"
echo "  qstat -x -u $USER                       # R=running; note the nodes"
echo "  cat /grand/FRAME-IDP/avasan/nemotron/ENDPOINT.txt   # head node + port once up"
echo "  tail -f ${JOBID%%.*}.OU                 # vLLM log; wait for 'Application startup complete'"
