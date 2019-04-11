#!/bin/bash
script cox.out

for imputation in mean forward
do
    for i in 1 21 23 30
    do
        python3 Cox.py --imputation_mode $imputation --seed $i --positive_weight 56
    done
done

exit
