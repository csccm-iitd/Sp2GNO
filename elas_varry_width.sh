#!/bin/bash

# Values for width
#width_list=(16 32 64 128 256)

width_list=( 64 128)

for width in "${width_list[@]}"; do
  echo "Running experiment with width=$width"
  python train_elas_ablation.py --width "$width" --exp_name "varry_width"
done
