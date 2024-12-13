#!/bin/bash

# Values for N
# k_list=(15 20 25 30 35)
k_list=(25 30 35)


# Fixed values for other parameters
experiment='varry_k'

for k in "${k_list[@]}"; do
  echo "Running experiment with k=$k"
  python3 train_elas_ablation.py  --k "$k" --exp_name "$experiment"
done
