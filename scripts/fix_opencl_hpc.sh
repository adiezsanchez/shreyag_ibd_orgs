#!/usr/bin/env bash
set -euo pipefail

echo "Checking OpenCL setup for HPC (Pixi env)..."

# Where Pixi puts the env
PIXI_ENV="${PIXI_PROJECT_ROOT:-$(pwd)}/.pixi/envs/default"

# libOpenCL.so: discover via ldconfig, else try known HPC paths (Idun, then Ubuntu)
OPENCL_LIB=$(ldconfig -p 2>/dev/null | grep -oE '/[^ ]+libOpenCL\.so\.1' | head -1)
if [ -z "$OPENCL_LIB" ]; then
  for p in /lib64/libOpenCL.so.1 /usr/local/cuda/targets/x86_64-linux/lib/libOpenCL.so.1 /usr/lib/x86_64-linux-gnu/libOpenCL.so.1; do
    [ -e "$p" ] && OPENCL_LIB="$p" && break
  done
fi
if [ -n "$OPENCL_LIB" ]; then
  ln -sf "$OPENCL_LIB" "$PIXI_ENV/lib/libOpenCL.so"
  echo "Linked libOpenCL.so -> $OPENCL_LIB"
else
  echo "WARN: libOpenCL.so.1 not found"
fi

# Ensure ICD vendor dir exists and link nvidia.icd
mkdir -p "$PIXI_ENV/etc/OpenCL/vendors"
if [ -e /etc/OpenCL/vendors/nvidia.icd ]; then
  ln -sf /etc/OpenCL/vendors/nvidia.icd "$PIXI_ENV/etc/OpenCL/vendors/nvidia.icd"
  echo "Linked nvidia.icd"
else
  echo "WARN: nvidia.icd not found at /etc/OpenCL/vendors/nvidia.icd"
fi

echo "OpenCL setup complete."
