#!/bin/bash


pip3 install torch --index-url https://download.pytorch.org/whl/cu118

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv  -f https://data.pyg.org/whl/torch-2.4.1+cu118.html

conda install -c conda-forge fenics=2019.1.0 -y

pip3 install -r requirements.txt
