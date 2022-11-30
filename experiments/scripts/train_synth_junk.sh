#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/train.py --dataset synth_junk \
   --save_freq 1000 \
   --checkpoint_freq 50000
