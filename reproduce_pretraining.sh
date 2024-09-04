#!/bin/bash

# This script trains and evaluates Hebbian semantic segmentation models based on unsupervised HPCA

set -e

REPS=1      # 1 because only pretraining
EVAL_GPU=0

EVAL_EXP_ROOT="./runs"
EVAL_DATA_ROOT="./data"

EXPS=(
    #################################
    # PH2 Dataset
    #################################
    ph2/hunet_base-hpca
    ph2/hunet_base-hpca_t
    ph2/vae-unet_base
    ph2/ddpm-unet_base
    #################################
    # GlaS Dataset
    #################################
    glas/hunet_base-hpca
    glas/hunet_base-hpca_t
    glas/vae-unet_base
    glas/ddpm-unet_base
    #################################
    # EYES Dataset
    #################################
    eyes/hunet_base-hpca
    eyes/hunet_base-hpca_t
    eyes/vae-unet_base
    eyes/ddpm-unet_base
)

# Train & Evaluate 
for EXP in ${EXPS[@]}; do
    HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.cross_val_bucket_validation_index=0 data.validation.cross_val_bucket_validation_index=0
done


# Test 
for EXP in ${EXPS[@]}; do
    case $EXP in 
        ph2*)
            CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-1/regime-1.0/run-0 --data-root $EVAL_DATA_ROOT/PH2 --in-memory True --best-on-metric last --output-file-name preds_from_last.csv;;
        eyes*)
            CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-1/regime-1.0/run-0 --data-root $EVAL_DATA_ROOT/NN_human_eyes --in-memory True --best-on-metric last --output-file-name preds_from_last.csv;;
        glas*)
            CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-1/regime-1.0/run-0 --data-root $EVAL_DATA_ROOT/GlaS/test --in-memory True --test-split all --best-on-metric last --output-file-name preds_from_last.csv;;
    esac
done
