#!/bin/bash
#SBATCH --account=mh-ikom
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=organoid_pipe_single

# Usage:
#   sbatch run_pipeline_slurm_single.sh <path_to_single_nd2> [config_path]
# Example:
#   sbatch run_pipeline_slurm_single.sh /cluster/work/albertds/raw_data/sample.nd2
#   sbatch run_pipeline_slurm_single.sh /cluster/work/albertds/raw_data/sample.nd2 config.yaml
#
# Processes a single .nd2 file on one A100 GPU (max 1 hour).
# Config path is optional; defaults to config.yaml (relative to project dir).

set -e

if [[ -z "${1:-}" ]]; then
  echo "Usage: sbatch run_pipeline_slurm_single.sh <path_to_single_nd2> [config_path]"
  exit 1
fi

ND2_PATH="$1"
CONFIG_PATH="${2:-config.yaml}"

PROJECT_DIR="/cluster/home/albertds/github_repos/shreyag_ibd_orgs"
cd "$PROJECT_DIR"

if [[ ! -f "$ND2_PATH" ]]; then
  echo "Error: file not found: $ND2_PATH"
  exit 1
fi

if [[ "$ND2_PATH" != *.nd2 ]]; then
  echo "Error: expected a .nd2 file, got: $ND2_PATH"
  exit 1
fi

echo "Processing single .nd2: $ND2_PATH"

pixi run fix_opencl

pixi run python tests/test_gpu.py

pixi run python main.py --image "$ND2_PATH" --config "$CONFIG_PATH"

echo "Done."
