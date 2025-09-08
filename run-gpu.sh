#!/bin/bash
# GPU-enabled run script for Windows/WSL

docker run --rm --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$(pwd)/data:/data" \
  automatic-transcriptions-gpu "$@"
