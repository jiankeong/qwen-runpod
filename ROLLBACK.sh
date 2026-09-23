#!/usr/bin/env bash
set -euo pipefail

root="${1:-$(pwd)}"
if [[ "${2:-}" == "--worker-readiness-only" ]]; then
  for path in handler.py tests/test_handler.py README.md; do
    git -C "$root" show "v1.0.11:$path" > "$root/$path"
  done
  printf 'ROLLBACK_OK: restored v1.0.11 early worker registration\n'
  exit 0
fi

if [[ "${2:-}" == "--test-gpu-4090-only" ]]; then
  for path in .runpod/tests.json README.md tests/test_runpod_hub.py; do
    git -C "$root" show "v1.0.10:$path" > "$root/$path"
  done
  printf 'ROLLBACK_OK: restored v1.0.10 A40 Hub test GPU\n'
  exit 0
fi

if [[ "${2:-}" == "--network-volume-only" ]]; then
  for path in Dockerfile handler.py .runpod/hub.json; do
    git -C "$root" show "v1.0.9:$path" > "$root/$path"
  done
  printf 'ROLLBACK_OK: restored v1.0.9 baked-model image\n'
  exit 0
fi

if [[ "${2:-}" == "--baked-model-only" ]]; then
  for path in Dockerfile handler.py .runpod/hub.json; do
    git -C "$root" show "v1.0.8:$path" > "$root/$path"
  done
  printf 'ROLLBACK_OK: restored v1.0.8 runtime model download\n'
  exit 0
fi

if [[ "${2:-}" == "--loader-path-only" ]]; then
  git -C "$root" show v1.0.7:Dockerfile > "$root/Dockerfile"
  printf 'ROLLBACK_OK: restored v1.0.7 Dockerfile without linker registration\n'
  exit 0
fi

if [[ "${2:-}" == "--test-pool-only" ]]; then
  python3 - "$root/.runpod/tests.json" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text().replace('"gpuTypeId": "NVIDIA A40"', '"gpuTypeId": "NVIDIA GeForce RTX 4090"', 1)
path.write_text(text)
print("ROLLBACK_OK: restored RTX 4090 Hub test pool")
PY
  exit 0
fi

if [[ "${2:-}" == "--mock-smoke-only" ]]; then
  for path in handler.py .runpod/tests.json .runpod/hub.json; do
    git -C "$root" show "v1.0.5:$path" > "$root/$path"
  done
  printf 'ROLLBACK_OK: restored v1.0.5 model-loading Hub test\n'
  exit 0
fi

if [[ "${2:-}" == "--startup-flow-only" ]]; then
  git -C "$root" show v1.0.4:handler.py > "$root/handler.py"
  git -C "$root" show v1.0.4:.runpod/tests.json > "$root/.runpod/tests.json"
  printf 'ROLLBACK_OK: restored v1.0.4 startup flow and Hub smoke-test model\n'
  exit 0
fi

if [[ "${2:-}" == "--test-gpu-only" ]]; then
  python3 - "$root/.runpod/tests.json" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text().replace('"gpuTypeId": "NVIDIA A40"', '"gpuTypeId": "NVIDIA GeForce RTX 4090"', 1)
path.write_text(text)
print("ROLLBACK_OK: restored previous Hub test GPU")
PY
  exit 0
fi

if [[ "${2:-}" == "--image-fix-only" ]]; then
  python3 - "$root/Dockerfile" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
text = text.replace(
    "ghcr.io/ggml-org/llama.cpp:full-cuda@sha256:1276d8b3f09fe1cb05b8d7369f64a5a9c15619fb05a61f3df1d261a4b53c96d9",
    "ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:0192ab2545efcbe79c240645e34abd8fffbe4813aedcef5a0e3a886ef6d6d82f",
    1,
)
path.write_text(text)
print("ROLLBACK_OK: restored previous server-cuda image reference")
PY
  exit 0
fi

for path in handler.py Dockerfile requirements.txt .dockerignore .env.example deploy.sh README.md changes.diff VERIFICATION.txt; do
  rm -f "$root/$path"
done
rm -rf "$root/tests" "$root/__pycache__"
rm -rf "$root/.runpod"
if [[ "$root" != "$(pwd)" ]]; then
  rm -f "$root/ROLLBACK.sh"
fi
printf 'ROLLBACK_OK: restored empty workspace\n'
