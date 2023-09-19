#!/bin/bash

set -e

REPS=5
START_REP=0
GPU=0
K_VALUES=(
    0.01
    0.02
    0.05
    0.1
    0.2
    0.5
    1
)

EXPS=(
    #################################
    # PH2 Dataset
    #################################
    ph2/hunet-swta
    ph2/hunet-swta_ft
    #ph2/hunet2-swta
    #ph2/hunet2-swta_ft
)

# Train & Evaluate (k-cross validation)
for K in ${K_VALUES[@]}; do
    for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do    # Multiple repetitions of the same experiment, varying the validation bucket
        for EXP in ${EXPS[@]}; do
            CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.cross_val_bucket_validation_index=$REP model.hebb.k=$K
        done
    done
done


# Test 
for K in ${K_VALUES[@]}; do
    for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
        for EXP in ${EXPS[@]}; do
            CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/inv_temp-$K/run-$REP --data-root data/PH2 --in-memory True --device cpu
        done
    done
done

