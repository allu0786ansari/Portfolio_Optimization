#!/usr/bin/env bash
# Install the locked dependencies for the given extras, with CPU-only PyTorch.
#
#   scripts/install_deps.sh "serving"                 # into the system Python (Docker)
#   PYTHON=/opt/venv/bin/python scripts/install_deps.sh "dev rl ml"
#
# Versions come from uv.lock (exported). The CUDA wheels that PyPI's Linux torch
# pulls in (~3 GB) are filtered out and torch is installed from the CPU index at
# the exact locked version, keeping images and CI fast.
set -euo pipefail
EXTRAS="${1:-}"
FLAGS=()
for e in $EXTRAS; do FLAGS+=(--extra "$e"); done
TARGET=(--system)
if [[ -n "${PYTHON:-}" ]]; then TARGET=(--python "$PYTHON"); fi

TMP="$(mktemp -d)"
uv export --frozen --no-hashes --no-dev --no-emit-project "${FLAGS[@]}" > "$TMP/all.txt"
grep -vE '^(torch|triton|nvidia-|cuda-)' "$TMP/all.txt" > "$TMP/req.txt"
TORCH="$(grep -E '^torch==' "$TMP/all.txt" | sed -E 's/^torch==([^ ;]+).*/\1/' || true)"
if [[ -n "$TORCH" ]]; then
  echo "Installing CPU torch==${TORCH}"
  uv pip install "${TARGET[@]}" --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH}"
fi
uv pip install "${TARGET[@]}" --no-deps -r "$TMP/req.txt"
rm -rf "$TMP"
