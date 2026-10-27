#!/bin/bash
# 03 - Launch the vLLM OpenAI server on GPUs via Cobalt.
# The actual job scripts live one level up:
#   serve_nemotron.sh       - fixed TP4, for the 4x H100 node (gpu_h100)
#   serve_nemotron_auto.sh  - auto-detects GPU count & picks TP, for any node (e.g. gpu_b200)
# gpu_h100 max walltime = 6h (360 min); gpu_b200 max = 8h (480 min).
set -euo pipefail
BASE=/vast/users/avasan/nemotron
cd /home/avasan   # submit from /home so Cobalt writes <jobid>.output/.error there

QUEUE=${1:-gpu_h100}
case "$QUEUE" in
  gpu_h100) WALL=360; SCRIPT=$BASE/serve_nemotron.sh ;;
  gpu_b200) WALL=480; SCRIPT=$BASE/serve_nemotron_auto.sh ;;
  *)        WALL=360; SCRIPT=$BASE/serve_nemotron_auto.sh ;;
esac

qsub -q "$QUEUE" -t "$WALL" -n 1 "$SCRIPT"
echo "Submitted to $QUEUE (walltime ${WALL}m). Track:"
echo "  qstat -u avasan                         # wait for State=running, note Location"
echo "  cat $BASE/ENDPOINT.txt                  # node/port once it launches"
echo "  tail -f /home/avasan/<jobid>.output     # startup log (look for 'Application startup complete')"
