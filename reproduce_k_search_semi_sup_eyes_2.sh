#!/bin/bash

# This script aims to search the best inv-temp hyperparameter concerning the Hebbian models 
# belonging to the SWTA paradigm (both unsupervised and semi-supervised) varying the regimes

set -e

REPS=1
START_REP=0
EVAL_GPU=5

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

REGIMES=(
    0.01
    0.02
    0.05
    0.1
    0.2
    #1.0
)

EXPS=(
    #################################
    # PH2 Dataset
    #################################
    #ph2/hunet_base-swta
    #ph2/hunet_base-swta_ft
    #ph2/hunet-swta_ft
    #ph2/hunet_base-swta_t
    #ph2/hunet_base-swta_t_ft
    #ph2/hunet-swta_t_ft
    #################################
    # GlaS Dataset
    #################################
    #glas/hunet_base-swta
    #glas/hunet_base-swta_ft
    #glas/hunet-swta_ft
    #glas/hunet_base-swta_t
    #glas/hunet_base-swta_t_ft
    #glas/hunet-swta_t_ft
    #################################
    # EYES Dataset
    #################################
    #eyes/hunet_base-swta
    #eyes/hunet_base-swta_ft
    #eyes/hunet-swta_ft
    #eyes/hunet_base-swta_t
    #eyes/hunet_base-swta_t_ft
    #eyes/hunet-swta_t_ft
    eyes/perturbed-hunet-swta_t_ft
    #eyes/teacher-hunet-swta_t_ft
    #eyes/wavelet-hunet-swta_t_ft
    
)

# Train & Evaluate (pretraining)
for K in ${K_VALUES[@]}; do
    for EXP in ${EXPS[@]}; do
        case $EXP in
            glas*)   # this dataset has a fixed test split
                case $EXP in
                    */*_ft)
                        ;;
                    *)
                        HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.cross_val_bucket_validation_index=0 model.hebb.k=$K;;
                esac;;
            *)
                case $EXP in
                    */*_ft)
                        ;;
                    *)
                        HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.cross_val_bucket_validation_index=0 data.validation.cross_val_bucket_validation_index=0 model.hebb.k=$K;;
                esac;;                
        esac
    done
done


# Train & Evaluate (fine-tuning)
for R in ${REGIMES[@]}; do
    for K in ${K_VALUES[@]}; do
        for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do    
            for EXP in ${EXPS[@]}; do
                case $EXP in
                    glas*)   # this dataset has a fixed test split
                        case $EXP in
                            */*_ft)
                                HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.cross_val_bucket_validation_index=$REP model.hebb.k=$K data.train.smpleff_regime=$R data.train.split_seed=$REP;;
                            *)
                                ;;
                        esac;;
                    *)
                        case $EXP in
                            */*_ft)
                                 HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.cross_val_bucket_validation_index=$REP data.validation.cross_val_bucket_validation_index=$REP model.hebb.k=$K data.train.smpleff_regime=$R;;
                            *)
                                ;;
                        esac;;                
                esac
            done
        done
    done
done


# Test (pretraining)
for K in ${K_VALUES[@]}; do
    for EXP in ${EXPS[@]}; do
        case $EXP in
            ph2*)
                case $EXP in
                    */*_ft)
                        ;;
                    *)
                        CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-$K/regime-1.0/run-0 --data-root $EVAL_DATA_ROOT/PH2 --in-memory True --best-on-metric last --output-file-name preds_from_last.csv;;
                esac;;
            eyes*)
                case $EXP in
                    */*_ft)
                        ;;
                    *)
                        CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-$K/regime-1.0/run-0 --data-root $EVAL_DATA_ROOT/NN_human_eyes --in-memory True --best-on-metric last --output-file-name preds_from_last.csv;;
                esac;;          
            glas*)  # this dataset has a fixed test split
                case $EXP in
                    */*_ft)
                        ;;
                    *)
                        CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-$K/regime-1.0/run-0 --data-root $EVAL_DATA_ROOT/GlaS/test --in-memory True --test-split all --best-on-metric last --output-file-name preds_from_last.csv;;
                esac
        esac
    done
done


# Test (finetuning)
for R in ${REGIMES[@]}; do
    for K in ${K_VALUES[@]}; do
        for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
            for EXP in ${EXPS[@]}; do
                case $EXP in
                    ph2*)
                        case $EXP in
                            */*_ft)
                                CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-$K/regime-$R/run-$REP --data-root $EVAL_DATA_ROOT/PH2 --in-memory True --best-on-metric dice;;
                            *)
                                ;;
                        esac;;
                    eyes*)
                        case $EXP in
                            */*_ft)
                                CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-$K/regime-$R/run-$REP --data-root $EVAL_DATA_ROOT/NN_human_eyes --in-memory True --best-on-metric dice;;
                            *)
                                ;;
                        esac;;    
                    glas*)  # this dataset has a fixed test split
                        case $EXP in
                            */*_ft)
                                CUDA_VISIBLE_DEVICES=$EVAL_GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-$K/regime-$R/run-$REP --data-root $EVAL_DATA_ROOT/GlaS/test --in-memory True --test-split all --best-on-metric dice;;
                            *)
                                ;;
                        esac
                esac
            done
        done
    done
done
