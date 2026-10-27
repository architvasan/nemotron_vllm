#!/bin/bash
# 02 - Build the vLLM apptainer image (SIF) from the official Docker image.
# WHY it's two stages:
#   * The login-node user cgroup is capped at 8 GB RAM, so `apptainer pull` /
#     mksquashfs get OOM-killed there.
#   * Rootless podman has no subuid range here, so it needs overlay +
#     ignore_chown_errors to unpack.
# So: podman pulls + exports a tarball on the LOGIN node (low memory, local /tmp),
# then apptainer builds the SIF in a short COMPUTE-node job (full memory, local disk).
set -euo pipefail

BASE=/vast/users/avasan/nemotron
IMAGE=docker://docker.io/vllm/vllm-openai:v0.32.0
OCITAR=$BASE/vllm-0.32.0.oci.tar
SIF=$BASE/vllm-0.32.0.sif

### Stage A (LOGIN node): pull image with podman, save to an OCI tarball on /vast
mkdir -p /tmp/podman-store /tmp/podman-run /tmp/podman-tmp
export TMPDIR=/tmp/podman-tmp
podman --root /tmp/podman-store --runroot /tmp/podman-run --storage-driver overlay \
  pull --storage-opt overlay.ignore_chown_errors=true "$IMAGE"
podman --root /tmp/podman-store --runroot /tmp/podman-run --storage-driver overlay \
  save --format oci-archive -o "$OCITAR" docker.io/vllm/vllm-openai:v0.32.0
echo "saved $(ls -lh "$OCITAR" | awk '{print $5}') -> $OCITAR"

### Stage B (COMPUTE node job): convert the tarball to a SIF where there is no 8 GB cap
cat > /home/avasan/nemo_build_sif.sh <<'EOF'
#!/bin/bash
set -ex
BASE=/vast/users/avasan/nemotron
export APPTAINER_TMPDIR=/tmp/apptbuild_$$      # node-local disk supports xattr/chown
export APPTAINER_CACHEDIR=/tmp/apptcache_$$
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
hostname; date
apptainer build "$BASE/vllm-0.32.0.sif" "oci-archive:$BASE/vllm-0.32.0.oci.tar"
ls -lh "$BASE/vllm-0.32.0.sif"
rm -rf "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
echo BUILD_DONE_OK
EOF
chmod +x /home/avasan/nemo_build_sif.sh

cd /home/avasan
qsub -q gpu_h100 -t 25 -n 1 /home/avasan/nemo_build_sif.sh
echo "SIF build submitted. Watch with: qstat -u avasan ; then: ls -lh $SIF"
echo "When done you can delete the tarball: rm -f $OCITAR"
