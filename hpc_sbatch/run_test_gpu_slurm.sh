#!/bin/bash
#SBATCH --account=mh-ikom
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:2
#SBATCH --time=0:10:00
#SBATCH --job-name=test_gpu_min

set -e
cd /cluster/home/albertds/github_repos/shreyag_ibd_orgs

pixi run fix_opencl
pixi run python tests/test_gpu.py
