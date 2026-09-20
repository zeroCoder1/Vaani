#!/usr/bin/env bash
# Upload the staged model to an S3-compatible bucket and verify it.
#
# Credentials are never passed here or read from the environment: rclone uses
# the remote you configured with `rclone config`, so the secret key stays in
# ~/.config/rclone/rclone.conf.
#
#   tools/dev.sh stage
#   tools/upload.sh <rclone-remote> <bucket> [public-base-url]
#
# e.g. tools/upload.sh r2 vaani https://pub-xxxx.r2.dev
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/models/upload"

REMOTE="${1:?usage: tools/upload.sh <rclone-remote> <bucket> [public-base-url]}"
BUCKET="${2:?usage: tools/upload.sh <rclone-remote> <bucket> [public-base-url]}"
PUBLIC="${3:-}"

command -v rclone >/dev/null || { echo "rclone not installed: brew install rclone"; exit 1; }
[ -f "$SRC/manifest.json" ] || { echo "run tools/dev.sh stage first"; exit 1; }

echo "uploading $(ls "$SRC" | wc -l | tr -d ' ') files ($(du -sh "$SRC" | cut -f1)) to $REMOTE:$BUCKET"
# --s3-chunk-size keeps multipart parts reasonable for the 878MB encoder.
rclone copy "$SRC" "$REMOTE:$BUCKET" \
  --progress \
  --transfers 4 \
  --s3-chunk-size 64M \
  --s3-upload-concurrency 4

echo
echo "remote listing:"
rclone size "$REMOTE:$BUCKET"

if [ -n "$PUBLIC" ]; then
  echo
  echo "verifying public access at $PUBLIC"
  code=$(curl -s -o /dev/null -w '%{http_code}' "$PUBLIC/manifest.json")
  echo "  manifest.json           HTTP $code"
  len=$(curl -sI "$PUBLIC/encoder.onnx.data" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r')
  echo "  encoder.onnx.data       ${len:-no content-length} bytes (expect 878444544)"
  ranges=$(curl -sI "$PUBLIC/encoder.onnx.data" | awk 'tolower($1)=="accept-ranges:"{print $2}' | tr -d '\r')
  echo "  accept-ranges           ${ranges:-absent} (bytes = resumable downloads)"
  [ "$code" = "200" ] && echo "  OK - use $PUBLIC as the model host" || echo "  bucket is not public yet"
fi
