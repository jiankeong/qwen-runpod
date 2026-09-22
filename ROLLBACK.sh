#!/usr/bin/env bash
set -euo pipefail

root="${1:-$(pwd)}"
for path in handler.py Dockerfile requirements.txt .dockerignore .env.example deploy.sh README.md changes.diff VERIFICATION.txt; do
  rm -f "$root/$path"
done
rm -rf "$root/tests" "$root/__pycache__"
rm -rf "$root/.runpod"
if [[ "$root" != "$(pwd)" ]]; then
  rm -f "$root/ROLLBACK.sh"
fi
printf 'ROLLBACK_OK: restored empty workspace\n'
