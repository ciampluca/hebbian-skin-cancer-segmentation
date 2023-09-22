#!/bin/bash

set -e

REPS=5
START_REP=0
GPU=0
REGIMES=(
    0.01
    0.02
    0.03
    0.04
    0.05
    0.10
    0.25
)

EXPS=(
    #################################
    # PH2 Dataset
    #################################
    ph2/unet
    ph2/hunet-swta_ft
    ph2/hunet-swta_t_ft
    ph2/hunet-hpca_ft
    ph2/hunet-hpca_t_ft
    ph2/hunet2-swta_ft
    ph2/hunet2-hpca_ft


)

# Train & Evaluate (k-cross validation)
for R in ${REGIMES[@]}; do
    for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do    # Multiple repetitions of the same experiment, varying the validation bucket
        for EXP in ${EXPS[@]}; do
            CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.cross_val_bucket_validation_index=$REP data.train.smpleff_regime=$R
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

