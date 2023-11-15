#!/bin/bash

# This script trains and evaluates standard semantic segmentation models used as baselines

set -e

REPS=3
START_REP=0
GPU=0

EVAL_EXP_ROOT="./runs"
EVAL_DATA_ROOT="./data"

EXPS=(
    #################################
    # PH2 Dataset
    #################################
    ph2/vae_unet
    ph2/vae_unet_ft
    ph2/vae_fcn32s
    ph2/vae_fcn32s_ft
    #################################
    # ISIC2016 Dataset
    #################################
    isic2016/vae_unet
    isic2016/vae_unet_ft
    isic2016/vae_fcn32s
    isic2016/vae_fcn32s_ft
    #################################
    # KvasirSEG Dataset
    #################################
    kvasirSEG/vae_unet
    kvasirSEG/vae_unet_ft
    kvasirSEG/vae_fcn32s
    kvasirSEG/vae_fcn32s_ft
    #################################
    # DataScienceBowl2018 Dataset
    #################################
    datasciencebowl2018/vae_unet
    datasciencebowl2018/vae_unet_ft
    datasciencebowl2018/vae_fcn32s
    datasciencebowl2018/vae_fcn32s_ft
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
        case $EXP in 
            ph2*)
                CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-1/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/PH2 --in-memory True;;
            isic2016*)
                CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-1/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/ISIC2016 --in-memory True;;
            kvasirSEG*)
                CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-1/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/KvasirSEG --in-memory True;;
            datasciencebowl2018*)
                CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-1/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/DataScienceBowl2018 --in-memory True;;
        esac
    done
done
