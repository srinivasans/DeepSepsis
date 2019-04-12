#!/bin/bash

########### Run LSTMM Experiments ###########

declare cell="LSTMM"

#Mean Imputation
for i in 1 21 23 30
do
    python3 sepsis_prediction.py --experiment=$cell --celltype=$cell --epochs=25 --seed=$i --data-path=data/full_data --early-stopping-patience=10 --calculate-delay=0
done


# Forward Imputation
for i in 1 21 23 30
do
    python3 sepsis_prediction.py --experiment=$cell --celltype=$cell --epochs=25 --seed=$i --data-path=data/full_data --early-stopping-patience=10 --impute-forward=1 --imputation-method=forward --calculate-delay=0
done


#DAE Imputation
for i in 1 21 23 30
do
    python3 sepsis_prediction.py --experiment=$cell --celltype=$cell --epochs=25 --seed=$i --data-path=data/DAE --imputation-folder=data/full_data --early-stopping-patience=10 --imputation-method=dae --calculate-delay=0
done

#GRUD Imputation
for i in 1 21 23 30
do
    nohup python3 sepsis_prediction.py --experiment=$cell --celltype=$cell --epochs=25 --seed=$i --data-path=data/challenge_data_GRUD_imputed --early-stopping-patience=10 --imputation-method=grud --calculate-delay=0 --imputation-folder=data/full_data > nohup"$cell""$i".out &
done

###########################################


########### Run LSTMSimple Experiments ###########

declare cell="LSTMSimple"

# Mean Imputation
for i in 1 21 23 30
do
    python3 sepsis_prediction.py --experiment=$cell --celltype=$cell --epochs=25 --seed=$i --data-path=data/full_data --early-stopping-patience=10
done


# Forward Imputation
for i in 1 21 23 30
do
    python3 sepsis_prediction.py --experiment=$cell --celltype=$cell --epochs=25 --seed=$i --data-path=data/full_data --early-stopping-patience=10 --impute-forward=1 --imputation-method=forward
done


#DAE Imputation
for i in 1 21 23 30
do
    python3 sepsis_prediction.py --experiment=$cell --celltype=$cell --epochs=25 --seed=$i --data-path=data/DAE --imputation-folder=data/full_data --early-stopping-patience=10 --imputation-method=dae
done

#GRUD Imputation
for i in 1 21 23 30
do
    nohup python3 sepsis_prediction.py --experiment=$cell --celltype=$cell --epochs=25 --seed=$i --data-path=data/challenge_data_GRUD_imputed --early-stopping-patience=10 --imputation-method=grud --imputation-folder=data/full_data > nohup"$cell""$i".out &
done

###########################################


########### Run LSTM Experiments ###########

declare cell="LSTM"

# Mean Imputation
for i in 1 21 23 30
do
    python3 sepsis_prediction.py --experiment=$cell --celltype=$cell --epochs=25 --seed=$i --data-path=data/full_data --early-stopping-patience=10 --calculate-delay=0
done


# Forward Imputation
for i in 1 21 23 30
do
    python3 sepsis_prediction.py --experiment=$cell --celltype=$cell --epochs=25 --seed=$i --data-path=data/full_data --early-stopping-patience=10 --impute-forward=1 --imputation-method=forward --calculate-delay=0
done


#DAE Imputation
for i in 1 21 23 30
do
    python3 sepsis_prediction.py --experiment=$cell --celltype=$cell --epochs=25 --seed=$i --data-path=data/DAE --early-stopping-patience=10 --imputation-method=dae --calculate-delay=0
done

#GRUD Imputation
for i in 1 21 23 30
do
    python3 sepsis_prediction.py --experiment=$cell --celltype=$cell --epochs=25 --seed=$i --data-path=data/challenge_data_GRUD_imputed --early-stopping-patience=10 --imputation-method=grud --calculate-delay=0
done

###########################################
