#!/usr/bin/env bash
# file: tensorboard.sh
#
# Start the TensorBoard and inspect training and validation performance.
#

set -e

tensorboard --logdir logs/
exit 0

# EOF
