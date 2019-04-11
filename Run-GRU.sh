#!/bin/bash

# Run GRU Experiments
# Mean Imputation
for cell in GRU GRUM
do
    for i in 1 21 23 30    
    do
        python3 sepsis_prediction.py --data-path=data/challenge_data --imputation-method=mean --celltype=$cell --experiment=$cell --seed=$i  
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

# # Run GRU Experiments
# DAE Imputation
for cell in GRU GRUM
do
    for i in 1 21 23 30    
    do
        python3 sepsis_prediction.py --data-path=data/challenge_data_DAE_imputed --imputation-method=dae --celltype=$cell --experiment="$cell"dae --seed=$i --imputation-folder=data/challenge_data
    done
done


# Run GRU Experiments
# GRUD Imputation
for cell in GRU GRUM
do
    for i in 1 21 23 30    
    do
        python3 sepsis_prediction.py --data-path=data/challenge_data_GRUD_imputed --imputation-method=grud --celltype=$cell --experiment="$cell"grud --seed=$i  --imputation-folder=data/challenge_data
    done
done