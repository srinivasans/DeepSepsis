#!/bin/bash
export CUDA_HOME=/pkgs_local/cuda-9.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pkgs_local/cuda-9.0/lib64
python3 Run-GRUD.py physionet sepsis --early_stopping_patience 25 --epochs 200 > run.out