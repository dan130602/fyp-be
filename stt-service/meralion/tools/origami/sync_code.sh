#!/bin/bash
# Copy the worker SOURCE to Origami scratch (run on the laptop, from stt-service/meralion). Bytecode caches are excluded. Prints the vendored-sources hash on both sides.
#   bash tools/origami/sync_code.sh [ssh alias, default origami-fyp]
set -euo pipefail
HOST=${1:-origami-fyp}
REMOTE=/common/scratch/users/e/YOUR_USERNAME/meralion/runs/worker-v0/code
PILOT_LIST=../../stt-experiments/meralion-runner/pilot_calls_overlap_v1.json
cd "$(dirname "$0")/../.."
test -f "$PILOT_LIST"
# the commit of the code being synced; a dirty tree is refused (an acceptance run never uses dirty source; ALLOW_DIRTY=1 only for non-acceptance diagnostics and appends -dirty to the recorded commit)
COMMIT=$(git rev-parse HEAD)
git update-index -q --refresh || true      # a stat-stale index entry (a CRLF working copy under core.autocrlf) must not be reported as a content change; the byte check below sees the real difference
STATUS=$(git status --porcelain -- .)
if [ -n "$STATUS" ]; then
  if [ "${ALLOW_DIRTY:-0}" = "1" ]; then COMMIT="$COMMIT-dirty"; echo "WARNING: dirty tree synced as $COMMIT (not usable for an acceptance run)"
  else echo "refusing to sync: stt-service/meralion has uncommitted changes:"; echo "$STATUS"; exit 4; fi
fi
SHIPPED="meralion_worker configs patches tools/apply_patches.py tools/env_identity.py tools/audio_fingerprint.py tools/make_jobs.py tools/gpu_scope.py tools/origami"
# REV 4.1 A1: what is shipped must be BYTE-IDENTICAL to the committed blobs (the tool hash in the plugin test's report is compared with the committed file; a CRLF working copy from core.autocrlf would differ).
# `git hash-object --no-filters` hashes the working-tree bytes as they are; it must equal the blob id in HEAD for every tracked shipped file. Checked before anything is sent; skipped only for ALLOW_DIRTY (-dirty is never acceptance).
if [ "$COMMIT" = "$(git rev-parse HEAD)" ]; then
  BAD=0
  while IFS=$'\t' read -r META FILE; do
    BLOB=${META##* }
    ACTUAL=$(git hash-object --no-filters -- "$FILE" 2>/dev/null) || ACTUAL="unreadable"
    if [ "$BLOB" != "$ACTUAL" ]; then echo "  not the committed bytes: $FILE (blob $BLOB, working tree $ACTUAL)"; BAD=$((BAD + 1)); fi
  done < <(git ls-tree -r HEAD -- $SHIPPED)
  if [ "$BAD" -ne 0 ]; then echo "refusing to sync: $BAD shipped file(s) differ byte-for-byte from the committed blobs (line endings? rewrite them from git, then retry)"; exit 6; fi
fi
ssh -o BatchMode=yes "$HOST" "rm -rf '$REMOTE' && mkdir -p '$REMOTE'"
tar --exclude='__pycache__' --exclude='*.pyc' -czf - $SHIPPED | ssh -o BatchMode=yes "$HOST" "tar -xzf - -C '$REMOTE'"
scp -q -o BatchMode=yes "$PILOT_LIST" "$HOST:$REMOTE/pilot_calls_overlap_v1.json"
printf '%s\n' "$COMMIT" | ssh -o BatchMode=yes "$HOST" "cat > '$REMOTE/CODE_COMMIT'"
echo "code commit recorded on the remote side: $(ssh -o BatchMode=yes "$HOST" "cat '$REMOTE/CODE_COMMIT'")"
echo "local : $(sha256sum meralion_worker/vendored/SOURCES.json | cut -d' ' -f1)"
echo "remote: $(ssh -o BatchMode=yes "$HOST" "sha256sum '$REMOTE/meralion_worker/vendored/SOURCES.json' | cut -d' ' -f1")"
# the patch files and the tool that applies them are compared on both sides (REV 3.0 A3): a difference stops the sync report with exit 5
PATCH_FILES="patches/manifest.json patches/vllm-0.16.0-mig-single-device.patch tools/apply_patches.py"
for f in $PATCH_FILES; do
  LOCAL_SUM=$(sha256sum "$f" | cut -d' ' -f1)
  REMOTE_SUM=$(ssh -o BatchMode=yes "$HOST" "sha256sum '$REMOTE/$f' | cut -d' ' -f1")
  echo "$f local $LOCAL_SUM remote $REMOTE_SUM"
  if [ "$LOCAL_SUM" != "$REMOTE_SUM" ]; then echo "MISMATCH: $f differs between the laptop and the cluster"; exit 5; fi
done
echo "bytecode files on the remote side (must be 0): $(ssh -o BatchMode=yes "$HOST" "find '$REMOTE' -name '*.pyc' -o -name __pycache__ | wc -l")"
