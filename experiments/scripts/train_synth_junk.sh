#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/train.py --dataset synth_junk \
   --save_freq 1000 \
   --checkpoint_freq 50000 \
   --refine_margin 0.018 \
   --start_epoch 5 \
   --resume_posenet /tools/DenseFusion/trained_models/synth_junk_v3/pose_model_4_0.021147409784266713.pth
