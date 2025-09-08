#!/bin/bash
# GPU-enabled run script for Windows/WSL

# Check if first argument is --entrypoint
if [ "$1" = "--entrypoint" ]; then
  # Handle --entrypoint case
  shift  # Remove --entrypoint
  ENTRYPOINT="$1"
  shift  # Remove the entrypoint command
  docker run --rm --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$(pwd)/data:/data" \
    --entrypoint "$ENTRYPOINT" \
    automatic-transcriptions-gpu "$@"
else
  # Normal case
  docker run --rm --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$(pwd)/data:/data" \
    automatic-transcriptions-gpu "$@"
fi
