#!/bin/bash

# Bagel Model Evaluation Script
#
# This script demonstrates how to run lmms-eval with Bagel's multimodal
# understanding pipeline, including the new interleaved visual reasoning loop.
#
# Prerequisites:
#   1. Install Bagel package:
#      uv pip install git+https://github.com/oscarqjh/Bagel.git
#
#   2. Download a trained checkpoint and pass the model path as $1.
#
# Usage:
#   bash examples/models/bagel.sh /path/to/BAGEL-7B-MoT mme

# Note: local checkpoint folder does not need to include inferencer.py when using the
# upstream Bagel package; inferencer is imported from the installed python package.

MODEL_PATH=$1
TASK=${2:-mme}

accelerate launch -m lmms_eval \
  --model bagel \
  --model_args pretrained=${MODEL_PATH},mode=understanding,reasoning_pipeline=interleaved,reasoning_max_iterations=8 \
  --tasks ${TASK} \
  --batch_size 1 \
  --log_samples \
  --output_path ./logs/
