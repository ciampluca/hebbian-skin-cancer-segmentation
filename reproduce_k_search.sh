#!/bin/bash

# This script aims to search the best inv-temp hyperparameter concerning the Hebbian models belonging to the SWTA paradigm

set -e

REPS=1
START_REP=0
EVAL_GPU=0

EVAL_EXP_ROOT="./runs"
EVAL_DATA_ROOT="./data"

K_VALUES=(
    1
    5
    10
    20
    50
    100
)

EXPS=(
    # #################################
    # # PH2 Dataset
    # #################################
    # ph2/hunet-swta
    # ph2/hunet-swta_ft
    # ph2/hunet-swta_t
    # ph2/hunet-swta_t_ft
    # #ph2/hunet2-swta
    # #ph2/hunet2-swta_ft
    # #################################
    # # ISIC2016 Dataset
    # #################################
    # isic2016/hunet-swta
    # isic2016/hunet-swta_ft
    # isic2016/hunet-swta_t
    # isic2016/hunet-swta_t_ft
    # #isic2016/hunet2-swta
    # #isic2016/hunet2-swta_ft
    # #################################
    # # KvasirSEG Dataset
    # #################################
    # kvasirSEG/hunet-swta
    # kvasirSEG/hunet-swta_ft
    # kvasirSEG/hunet-swta_t
    # kvasirSEG/hunet-swta_t_ft
    # #kvasirSEG/hunet2-swta
    # #kvasirSEG/hunet2-swta_ft
    # #################################
    # # DataScienceBowl2018 Dataset
    # #################################
    # datasciencebowl2018/hunet-swta
    # datasciencebowl2018/hunet-swta_ft
    # datasciencebowl2018/hunet-swta_t
    # datasciencebowl2018/hunet-swta_t_ft
    # #datasciencebowl2018/hunet2-swta
    # #datasciencebowl2018/hunet2-swta_ft
    #################################
    # GlaS Dataset
    #################################
    glas/hunet-swta
    glas/hunet-swta_ft
    glas/hunet_base-swta_ft
    glas/hunet-swta_t
    glas/hunet-swta_t_ft
    glas/hunet_base-swta_t_ft
    #glas/hfcn32s-swta
    #glas/hfcn32s-swta_ft
    #glas/hfcn32s_base-swta_ft
)

# Train & Evaluate (k-cross validation)
for K in ${K_VALUES[@]}; do
    for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do    # Multiple repetitions of the same experiment, varying the validation bucket
        for EXP in ${EXPS[@]}; do
            case $EXP in
                glas*)
                    if [[ $REP -lt 1 ]]; then    # this dataset has a fixed test split
                        HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.cross_val_bucket_validation_index=0 model.hebb.k=$K
                    fi;;
                *)
                    HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.cross_val_bucket_validation_index=$REP model.hebb.k=$K;;
            esac
        done
    done
done


# Test 
for K in ${K_VALUES[@]}; do
    for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
        for EXP in ${EXPS[@]}; do
            case $EXP in 
                ph2*)
                    CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-$K/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/PH2 --in-memory True;;
                isic2016*)
                    CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-$K/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/ISIC2016 --in-memory True;;
                kvasirSEG*)
                    CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-$K/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/KvasirSEG --in-memory True;;
                datasciencebowl2018*)
                    CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-$K/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/DataScienceBowl2018 --in-memory True;;
                glas*)
                    if [ $REP -lt 1 ]; then        # this dataset has a fixed test split
                        case $EXP in
                            */*_ft)
                                CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-$K/regime-1.0/run-0 --data-root $EVAL_DATA_ROOT/GlaS/test --in-memory True --test-split all;;
                            *)
                                CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-$K/regime-1.0/run-0 --data-root $EVAL_DATA_ROOT/GlaS/test --in-memory True --test-split all --best-on-metric last;;
                        esac
                    fi;;
            esac
        done
    done
done
