#!/bin/bash

# Values for num_wavelet_layers
#num_wavelet_layers_list=(2 4 6 8 10)
num_wavelet_layers_list=( 8 10)
experiment='varry_num_layers'

for num_wavelet_layers in "${num_wavelet_layers_list[@]}"; do
  echo "Running experiment with num_wavelet_layers=$num_wavelet_layers"
  python train_elas_ablation.py --num_wavelet_layers "$num_wavelet_layers" --exp_name "$experiment"
done
