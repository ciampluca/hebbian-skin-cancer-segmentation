#!/bin/bash

# This script aims to search the best inv-temp hyperparameter concerning the Hebbian models belonging to the SWTA paradigm

set -e

REPS=1
START_REP=0
GPU=0
K_VALUES=(
    0.001
    0.002
    0.005
    2
    5
    10
)

EXPS=(
    #################################
    # PH2 Dataset
    #################################
    ph2/hunet-swta
    ph2/hunet-swta_ft
    ph2/hunet-swta_t
    ph2/hunet-swta_t_ft
    #ph2/hunet2-swta
    #ph2/hunet2-swta_ft
    #################################
    # ISIC2016 Dataset
    #################################
    isic2016/hunet-swta
    isic2016/hunet-swta_ft
    isic2016/hunet-swta_t
    isic2016/hunet-swta_t_ft
    #isic2016/hunet2-swta
    #isic2016/hunet2-swta_ft
    #################################
    # KvasirSEG Dataset
    #################################
    kvasirSEG/hunet-swta
    kvasirSEG/hunet-swta_ft
    kvasirSEG/hunet-swta_t
    kvasirSEG/hunet-swta_t_ft
    #kvasirSEG/hunet2-swta
    #kvasirSEG/hunet2-swta_ft
    #################################
    # DataScienceBowl2018 Dataset
    #################################
    datasciencebowl2018/hunet-swta
    datasciencebowl2018/hunet-swta_ft
    datasciencebowl2018/hunet-swta_t
    datasciencebowl2018/hunet-swta_t_ft
    #datasciencebowl2018/hunet2-swta
    #datasciencebowl2018/hunet2-swta_ft
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #################################
    # BrainMRI Dataset
    #################################
    # brainMRI/hunet-swta
    # brainMRI/hunet-swta_ft
    # brainMRI/hunet-swta_t
    # brainMRI/hunet-swta_t_ft
    # #brainMRI/hunet2-swta
    # #brainMRI/hunet2-swta_ft
    #################################
    # DRIVE Dataset
    #################################
    # drive/hunet-swta
    # drive/hunet-swta_ft
    # drive/hunet-swta_t
    # drive/hunet-swta_t_ft
    # #drive/hunet2-swta
    # #drive/hunet2-swta_ft
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
            case $EXP in 
                ph2*)
                    CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/inv_temp-$K/regime-1.0/run-$REP --data-root data/PH2 --in-memory True;;
                isic2016*)
                    CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/inv_temp-$K/regime-1.0/run-$REP --data-root data/ISIC2016 --in-memory True;;
                kvasirSEG*)
                    CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/inv_temp-$K/regime-1.0/run-$REP --data-root data/KvasirSEG --in-memory True;;
                datasciencebowl2018*)
                    CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/inv_temp-$K/regime-1.0/run-$REP --data-root data/DataScienceBowl2018 --in-memory True;;
                # brainMRI*)
                #     CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/inv_temp-$K/regime-1.0/run-$REP --data-root data/BrainMRI --in-memory False;;
                # drive*)
                #     CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/inv_temp-$K/regime-1.0/run-$REP --data-root data/DRIVE --in-memory True;;
            esac
        done
    done
done
