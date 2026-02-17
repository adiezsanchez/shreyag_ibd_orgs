#!/bin/bash
#SBATCH --account=mh-ikom
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=organoid_pipe

# Usage:
#   sbatch run_pipeline_slurm.sh <directory_with_nd2> [config_path]
# Example:
#   sbatch run_pipeline_slurm.sh /cluster/work/albertds/raw_data/20260114_T7_2microns
#   sbatch run_pipeline_slurm.sh /cluster/work/albertds/raw_data/20260114_T7_2microns config.yaml
#
# Processes all .nd2 files in the directory in parallel across 4 GPUs.
# Config path is optional; defaults to config.yaml (relative to project dir).

set -e

if [[ -z "${1:-}" ]]; then
  echo "Usage: sbatch run_pipeline_slurm.sh <directory_with_nd2> [config_path]"
  exit 1
fi

INPUT_DIR="$1"
CONFIG_PATH="${2:-config.yaml}"
NGPUS=4

PROJECT_DIR="/cluster/home/albertds/github_repos/shreyag_ibd_orgs"
cd "$PROJECT_DIR"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: directory not found: $INPUT_DIR"
  exit 1
fi

# Collect .nd2 files
shopt -s nullglob
files=( "$INPUT_DIR"/*.nd2 )
shopt -u nullglob

if [[ ${#files[@]} -eq 0 ]]; then
  echo "Error: no .nd2 files found in $INPUT_DIR"
  exit 1
fi

echo "Found ${#files[@]} .nd2 file(s). Processing on $NGPUS GPU(s)."

pixi run fix_opencl

# Split work across GPUs: each GPU gets a chunk of the file list
n=${#files[@]}
chunk=$(( (n + NGPUS - 1) / NGPUS ))

for gpu in $(seq 0 $((NGPUS - 1))); do
  start=$(( gpu * chunk ))
  end=$(( start + chunk ))
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    for (( i = start; i < end && i < n; i++ )); do
      pixi run python main.py --image "${files[i]}" --config "$CONFIG_PATH"
    done
  ) &
done
wait

echo "Done."
