#!/bin/bash

# This script trains and evaluates standard semantic segmentation models used as baselines

set -e

REPS=1
START_REP=0
EVAL_GPU=0

EVAL_EXP_ROOT="./runs"
EVAL_DATA_ROOT="./data"

EXPS=(
    #################################
    # PH2 Dataset
    #################################
    # ph2/unet
    # ph2/fcn32s
    # #################################
    # # ISIC2016 Dataset
    # #################################
    # isic2016/unet
    # isic2016/fcn32s
    # #################################
    # # KvasirSEG Dataset
    # #################################
    # kvasirSEG/unet
    # kvasirSEG/fcn32s
    # #################################
    # # DataScienceBowl2018 Dataset
    # #################################
    # datasciencebowl2018/unet
    # datasciencebowl2018/fcn32s
    # #################################
    # # GlaS Dataset
    # #################################
    glas/unet
    # glas/fcn32s
)

# Train & Evaluate (k-cross validation)
for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do    # Multiple repetitions of the same experiment, varying the validation bucket
    for EXP in ${EXPS[@]}; do
        HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.cross_val_bucket_validation_index=$REP
    done
done


# Test 
for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
    for EXP in ${EXPS[@]}; do
        case $EXP in 
            ph2*)
                CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-1/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/PH2 --in-memory True;;
            isic2016*)
                CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-1/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/ISIC2016 --in-memory True;;
            kvasirSEG*)
                CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-1/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/KvasirSEG --in-memory True;;
            datasciencebowl2018*)
                CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-1/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/DataScienceBowl2018 --in-memory True;;
            glas*)
                CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-1/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/GlaS/test --in-memory True --test-split all;;
        esac
    done
done
