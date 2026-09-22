#!/usr/bin/env bash
set -euo pipefail

: "${IMAGE:?Set IMAGE, for example ghcr.io/you/qwen38-runpod:latest}"
docker build --platform linux/amd64 -t "$IMAGE" .
docker push "$IMAGE"
printf 'Pushed %s\n' "$IMAGE"
