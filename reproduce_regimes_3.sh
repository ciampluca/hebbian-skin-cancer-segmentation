#!/bin/bash

# This script trains and evaluates Hebbian semantic segmentation models based on semi-supervised HPCA varying the regime 
# and supervised-only concerning the baselines

set -e

REPS=10
START_REP=0
EVAL_GPU=0

EVAL_EXP_ROOT="./runs"
EVAL_DATA_ROOT="./data"

INV_TEMP_PH2=5        # to be set accordingly, used by SWTA
INV_TEMP_GlaS=15         # to be set accordingly, used by SWTA

REGIMES=(
    0.01
    0.02
    0.05
    0.1
    0.2
    #0.25
    #0.5
    #0.75
    #1.0
)

EXPS=(
    #################################
    # PH2 Dataset
    #################################
    #ph2/unet_base
    #ph2/unet
    #ph2/hunet_base-hpca_ft
    #ph2/hunet_base-hpca_t_ft
    #ph2/hunet-hpca_ft
    #ph2/hunet-hpca_t_ft
    #ph2/vae-unet_base_ft
    #ph2/vae-unet_ft
    #ph2/teacher-unet
    #################################
    # GlaS Dataset
    #################################
    #glas/unet_base
    #glas/unet
    #glas/hunet_base-hpca_ft
    #glas/hunet_base-hpca_t_ft
    #glas/hunet-hpca_ft
    #glas/hunet-hpca_t_ft
    #glas/vae-unet_base_ft
    #glas/vae-unet_ft
    #glas/teacher-unet
    #################################
    # TREND Dataset
    #################################
    #trend/unet_base
    #trend/unet
    #trend/hunet_base-hpca_ft
    #trend/hunet_base-hpca_t_ft
    #trend/hunet-hpca_ft
    #trend/hunet-hpca_t_ft
    #trend/vae-unet_base_ft
    #trend/vae-unet_ft
    eyes/teacher-unet
)

# Train & Evaluate
for R in ${REGIMES[@]}; do
    for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
        for EXP in ${EXPS[@]}; do
            case $EXP in
                glas*)  # this dataset has a fixed test split
                    HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.cross_val_bucket_validation_index=$REP data.train.smpleff_regime=$R data.train.split_seed=$REP;;
                trend*)     # regime 0.01 is skipped for this dataset since it is too small
                    if [ "$(echo "$R > 0.01" | bc)" = 1 ]; then
                        HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.cross_val_bucket_validation_index=$REP data.validation.cross_val_bucket_validation_index=$REP data.train.smpleff_regime=$R
                    fi;;
                *)
                    HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.cross_val_bucket_validation_index=$REP data.validation.cross_val_bucket_validation_index=$REP data.train.smpleff_regime=$R;;
            esac
        done
    done
done


# Test 
for R in ${REGIMES[@]}; do
    for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
        for EXP in ${EXPS[@]}; do
            case $EXP in 
                ph2*)
                    CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-1/regime-$R/run-$REP --data-root $EVAL_DATA_ROOT/PH2 --in-memory True --best-on-metric dice;;
                glas*)
                    CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-1/regime-$R/run-$REP --data-root $EVAL_DATA_ROOT/GlaS/test --in-memory True --test-split all --best-on-metric dice;;
                eyes*)
                    CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-1/regime-$R/run-$REP --data-root $EVAL_DATA_ROOT/NN_human_eyes --in-memory True --best-on-metric dice;;
            esac
        done
    done
done
