#!/bin/bash

set -e

REPS=5
START_REP=0
GPU=0

EXPS=(
    #################################
    # PH2 Dataset
    #################################
    ph2/unet
    ph2/hunet-hpca
    ph2/hunet-hpca_ft
    #ph2/hunet2-hpca
    #ph2/hunet2-hpca_ft
    brainMRI/unet
    brainMRI/hunet-hpca
    brainMRI/hunet-hpca_ft
    #brainMRI/hunet2-hpca
    #brainMRI/hunet2-hpca_ft
)

# Train & Evaluate (k-cross validation)
for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do    # Multiple repetitions of the same experiment, varying the validation bucket
    for EXP in ${EXPS[@]}; do
        CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.cross_val_bucket_validation_index=$REP
    done
done


# Test 
for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
    for EXP in ${EXPS[@]}; do
        CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/inv_temp-1/run-$REP --data-root data/PH2 --in-memory True
        CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/inv_temp-1/run-$REP --data-root data/BrainRMI --in-memory False
    done
done
