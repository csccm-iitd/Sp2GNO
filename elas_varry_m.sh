#!/bin/bash

# Values for num_of_frequency_to_learn
# frequency_list=(4 8 16 32 64)
frequency_list=( 32 64)
experiment='varry_m'

for num_of_frequency_to_learn in "${frequency_list[@]}"; do
  echo "Running experiment with num_of_frequency_to_learn=$num_of_frequency_to_learn"
  python3  train_elas_ablation.py --num_of_frequency_to_learn "$num_of_frequency_to_learn" --exp_name "$experiment"
done
