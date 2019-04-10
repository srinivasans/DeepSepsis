#!/bin/bash
for imputation in decay mean forward
do
    for i in 1 21 23 30
    do
        sbatch --partition=gpunodes --mail-type=ALL,TIME_LIMIT_90 --mail-user=$1 --output="$imputation"_seed_"$i".out Run-GRUD.sh $imputation $i
    done
done
