#!/bin/bash
export CUDA_HOME=/pkgs_local/cuda-9.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pkgs_local/cuda-9.0/lib64
python3 Run-GRUD.py physionet sepsis --early_stopping_patience 25 --epochs 200 > run.out

# Run GRU Experiments
# Mean Imputation
for cell in GRU GRUM GRUD
do
    for i in 1 21 23 30    
    do
        python3 sepsis_prediction.py --data-path=data/train --imputation-method=mean --celltype=$cell --experiment=$cell --seed=$i  
    done
done

# Run GRU Experiments
# Forward Imputation
for cell in GRU GRUM
do
    for i in 1 21 23 30    
    do
        python3 sepsis_prediction.py --data-path=data/challenge_data --imputation-method=forward --celltype=$cell --experiment=$cell --seed=$i  
    done
done

# Run GRU Experiments
# DAE Imputation
for cell in GRU GRUM
do
    for i in 1 21 23 30    
    do
        python3 sepsis_prediction.py --data-path=data/dae_imputed_data --imputation-method=dae --celltype=$cell --experiment=$cell --seed=$i  
    done
done