#!/bin/bash
export CUDA_HOME=/pkgs_local/cuda-9.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pkgs_local/cuda-9.0/lib64
python3 Run-GRUD.py physionet sepsis --data challenge_data --impute_forward 0 --seed $2 --early_stopping_patience 10 --imputation_mode $1 --epochs 200
