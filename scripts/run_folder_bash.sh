#!/usr/bin/env bash
set -euo pipefail
ImageFolder="${1:?Usage: $0 <image_folder> [config.yaml]}"
Config="${2:-config.yaml}"
shopt -s nullglob
files=( "$ImageFolder"/*.nd2 "$ImageFolder"/*.czi )
shopt -u nullglob
if [[ ${#files[@]} -eq 0 ]]; then
  echo "No .nd2 or .czi files found in: $ImageFolder"
  exit 0
fi
echo "Found ${#files[@]} image(s). Processing..."
n=0
for f in "${files[@]}"; do
  n=$((n + 1))
  echo "[$n/${#files[@]}] Processing: $(basename "$f")"
  pixi run python main.py --image "$f" --config "$Config"
done
echo "Finished ${#files[@]} image(s)."
