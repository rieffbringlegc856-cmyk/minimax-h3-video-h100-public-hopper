#!/bin/bash
set -e

# Архитектура CUDA под H100 SXM / H200 (Hopper sm_90)
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_ARCH="90"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd /workspace/ComfyUI
exec python -u rp_handler.py
