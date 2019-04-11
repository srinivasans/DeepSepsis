#!/bin/bash

for imputation in mean forward
do
    for i in 1 21 23 30
    do
        python3 Cox.py --imputation_mode $imputation --seed $i --positive_weight 56 | tee COX-"$imputation"_seed_"$i".out
    done
done
