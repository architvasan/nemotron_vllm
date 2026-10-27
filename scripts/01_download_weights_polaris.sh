#!/bin/bash
# 01 (Polaris) - Download Nemotron-3.5-Super-VL-120B weights (~249 GB) to /grand.
# Run on a Polaris LOGIN node (polaris-login-NN): login nodes have outbound
# internet via the ALCF proxy; COMPUTE nodes do NOT, so the weights must be on
# disk before the serve job starts.
#
# WHY these choices (Polaris-specific, differ from the JLSE recipe):
#   * Everything lives on /grand (FRAME-IDP): /home is small and, more
#     importantly, the serve job reads the weights from a project filesystem the
#     compute nodes mount. /grand => PBS needs `-l filesystems=home:grand`.
#   * System python3 is 3.6 (too old) and the user python3.11 has a broken
#     ensurepip, so the venv is built from the `conda/2026-10-01` module's
#     python (3.13, working venv+pip). We call it by absolute path because the
#     `conda` shell function is not defined in a non-interactive shell.
#   * Repo is public (no HF token needed) -- verified config.json -> 307, a
#     shard -> 302, both redirect to the CDN without auth.
set -euo pipefail

BASE=/grand/FRAME-IDP/avasan/nemotron
REPO=nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16
CONDA_PY=/soft/applications/conda/2026-10-01/mconda3/bin/python   # python 3.13

mkdir -p "$BASE"/{cache,tmp,model,logs}

# Login-node internet is via the ALCF proxy; make sure it is set.
export HTTPS_PROXY=${HTTPS_PROXY:-http://proxy.alcf.anl.gov:3128}
export HTTP_PROXY=${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}
export https_proxy=${https_proxy:-http://proxy.alcf.anl.gov:3128}
export http_proxy=${http_proxy:-http://proxy.alcf.anl.gov:3128}

# One-time venv with the HF CLI (+ hf_xet for fast Xet transfer).
if [ ! -x "$BASE/venv/bin/hf" ]; then
  "$CONDA_PY" -m venv "$BASE/venv"
  "$BASE/venv/bin/python" -m pip install --quiet --upgrade pip
  "$BASE/venv/bin/pip" install --quiet "huggingface_hub[cli]" hf_xet
fi

export HF_HOME="$BASE/cache"
export TMPDIR="$BASE/tmp"

# Long download -- run it in a detached tmux so it survives your SSH session.
# (A heavy multi-hour process left in the foreground of a login shell can be
# reaped; tmux keeps it alive and lets you reattach to watch.)
rm -f "$BASE/logs/DOWNLOAD_DONE"
tmux kill-session -t nemodl 2>/dev/null || true
tmux new-session -d -s nemodl \
  "export HF_HOME=$BASE/cache TMPDIR=$BASE/tmp \
     HTTPS_PROXY=$HTTPS_PROXY HTTP_PROXY=$HTTP_PROXY https_proxy=$https_proxy http_proxy=$http_proxy; \
   $BASE/venv/bin/hf download $REPO --local-dir $BASE/model > $BASE/logs/hf_download.log 2>&1; \
   echo EXIT=\$? >> $BASE/logs/hf_download.log; touch $BASE/logs/DOWNLOAD_DONE"

echo "Download started in tmux session 'nemodl'. Watch it with:"
echo "  tail -f $BASE/logs/hf_download.log"
echo "  tmux attach -t nemodl        # Ctrl-b d to detach"
echo "Done when $BASE/logs/DOWNLOAD_DONE appears. Then verify:"
echo "  ls $BASE/model/model-*-of-*.safetensors | wc -l   # expect 63"
echo "  du -sh $BASE/model                                 # expect ~249 GB"
