#!/bin/bash
# Record what the EC2 test host and the Docker image really are (REV 3.0 B3/B5), before any measured run. Reads, installs and changes nothing. Run on the instance as the Docker user:
#   bash record_instance.sh <out dir> [<image to inspect>]
# It writes <out dir>/instance.txt (key=value, facts and the checks below) and the raw outputs next to it, and exits 1 when a REQUIRED check fails (architecture, cgroup v2), 0 otherwise.
# The IMDSv2 facts (instance id, type, AZ, AMI) come from the instance metadata service; the ECS-registered memory (`aws ecs describe-container-instances`) is read by the teammate and recorded beside it.
# Environment checks that need the container (the portable environment comparison, REV 3.2 Amendment 7) are NOT done here: run tools/env_identity.py --check inside the container.
# Overrides for tests: UNAME_M, CGROUP_FS_TYPE, IMDS_BASE.
set -uo pipefail
OUT=${1:?out dir}
IMAGE=${2:-}
mkdir -p "$OUT"
FILE="$OUT/instance.txt"
: > "$FILE"
note() { echo "$1=$2" >> "$FILE"; }
run_to() { local name=$1; shift; "$@" > "$OUT/$name.txt" 2>&1 || echo "(exit $?)" >> "$OUT/$name.txt"; }

REQUIRED_OK=1
ARCH=${UNAME_M:-$(uname -m)}
note architecture "$ARCH"
if [ "$ARCH" = "x86_64" ]; then note architecture_ok yes; else note architecture_ok "NO (x86_64 required: the image and the pinned wheels are linux/amd64)"; REQUIRED_OK=0; fi
note kernel "$(uname -r)"
CGFS=${CGROUP_FS_TYPE:-$(stat -fc %T /sys/fs/cgroup 2>/dev/null || echo unknown)}
note cgroup_fs_type "$CGFS"
if [ "$CGFS" = "cgroup2fs" ]; then note cgroup_v2 yes; else note cgroup_v2 "NO (cgroup v2 is expected; memory.peak and memory.events are read from it)"; REQUIRED_OK=0; fi
note nproc "$(nproc 2>/dev/null || echo unknown)"
run_to host-free free -m
awk '/^Mem:/ {print "mem_total_mib=" $2; print "mem_available_mib=" $7} /^Swap:/ {print "swap_total_mib=" $2; print "swap_used_mib=" $3}' "$OUT/host-free.txt" >> "$FILE"

# instance metadata (IMDSv2); unavailable off EC2 is recorded as such, not guessed
IMDS=${IMDS_BASE:-http://169.254.169.254/latest}
TOKEN=$(curl -s -m 3 -X PUT "$IMDS/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || true)
imds() { if [ -n "$TOKEN" ]; then curl -s -m 3 -H "X-aws-ec2-metadata-token: $TOKEN" "$IMDS/meta-data/$1" 2>/dev/null || echo unavailable; else echo unavailable; fi; }
note instance_id "$(imds instance-id)"
note instance_type "$(imds instance-type)"
note availability_zone "$(imds placement/availability-zone)"
note ami_id "$(imds ami-id)"

# GPU, driver, container tooling
run_to nvidia-smi nvidia-smi
run_to nvidia-smi-query nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
note gpu_query "$(head -1 "$OUT/nvidia-smi-query.txt")"
note driver_cuda_level "$(grep -o 'CUDA Version: [0-9.]*' "$OUT/nvidia-smi.txt" | head -1)"
note cuda_visible_devices_on_host "${CUDA_VISIBLE_DEVICES:-unset}"
run_to docker-version docker version --format '{{.Server.Version}}'
note docker_server_version "$(head -1 "$OUT/docker-version.txt")"
run_to nvidia-ctk nvidia-ctk --version
note nvidia_container_toolkit "$(head -1 "$OUT/nvidia-ctk.txt")"
if [ -n "$IMAGE" ]; then
  IMG_ARCH=$(docker image inspect --format '{{.Architecture}}' "$IMAGE" 2>/dev/null || echo unavailable)
  note image "$IMAGE"
  note image_architecture "$IMG_ARCH"
  if [ "$IMG_ARCH" = "amd64" ]; then note image_architecture_ok yes; else note image_architecture_ok "NO (amd64 required)"; REQUIRED_OK=0; fi
fi
note required_checks_ok "$([ "$REQUIRED_OK" = 1 ] && echo yes || echo NO)"
cat "$FILE"
[ "$REQUIRED_OK" = 1 ]
