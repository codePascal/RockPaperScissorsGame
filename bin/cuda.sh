#!/usr/bin/env bash
# file: cuda.sh
#
# Verify proper functionality of CUDA.
#

poetry run python -c "import torch; print(torch.__version__); \
  print('CUDA:', torch.version.cuda); \
  print('available:', torch.cuda.is_available()); \
  print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
