#!/bin/bash

# This script trains and evaluates Hebbian semantic segmentation models

# Concerning Hebbian models belonging to the SWTA paradigm, it is needed to set the best temperature value found in the reproduce_k_search script

set -e

REPS=1
START_REP=0
GPU=0

EVAL_EXP_ROOT="./runs"
EVAL_DATA_ROOT="./data"

INV_TEMP_PH2=5        # to be set accordingly, used by SWTA
INV_TEMP_ISIC2016=2        # to be set accordingly, used by SWTA
INV_TEMP_KvasirSEG=5        # to be set accordingly, used by SWTA
INV_TEMP_DataScienceBowl2018=10        # to be set accordingly, used by SWTA

EXPS=(
    #################################
    # PH2 Dataset
    #################################
    # ph2/hunet-hpca
    # ph2/hunet-hpca_ft
    # ph2/hunet-hpca_t
    # ph2/hunet-hpca_t_ft
    ph2/hfcn32s-hpca
    ph2/hfcn32s-hpca_ft
    ph2/hfcn32s-hpca_t
    ph2/hfcn32s-hpca_t_ft
    # #ph2/hunet2-hpca_ft
    # #ph2/hunet2-hpca_t_ft
    # ph2/hunet-swta
    # ph2/hunet-swta_ft
    # ph2/hunet-swta_t
    # ph2/hunet-swta_t_ft
    ph2/hfcn32s-swta
    ph2/hfcn32s-swta_ft
    ph2/hfcn32s-swta_t
    ph2/hfcn32s-swta_t_ft
    #ph2/hunet2-swta_ft
    #ph2/hunet2-hpca_ft
    #################################
    # ISIC2016 Dataset
    #################################
    # isic2016/hunet-hpca
    # isic2016/hunet-hpca_ft
    # isic2016/hunet-hpca_t
    # isic2016/hunet-hpca_t_ft
    isic2016/hfcn32s-hpca
    isic2016/hfcn32s-hpca_ft
    isic2016/hfcn32s-hpca_t
    isic2016/hfcn32s-hpca_t_ft
    #isic2016/hunet2-hpca_ft
    #isic2016/hunet2-hpca_t_ft
    # isic2016/hunet-swta
    # isic2016/hunet-swta_ft
    # isic2016/hunet-swta_t
    # isic2016/hunet-swta_t_ft
    isic2016/hfcn32s-swta
    isic2016/hfcn32s-swta_ft
    isic2016/hfcn32s-swta_t
    isic2016/hfcn32s-swta_t_ft
    #isic2016/hunet2-swta_ft
    #isic2016/hunet2-hpca_ft
    #################################
    # KvasirSEG Dataset
    #################################
    # kvasirSEG/hunet-hpca
    # kvasirSEG/hunet-hpca_ft
    # kvasirSEG/hunet-hpca_t
    # kvasirSEG/hunet-hpca_t_ft
    kvasirSEG/hfcn32s-hpca
    kvasirSEG/hfcn32s-hpca_ft
    kvasirSEG/hfcn32s-hpca_t
    kvasirSEG/hfcn32s-hpca_t_ft
    #kvasirSEG/hunet2-hpca_ft
    #kvasirSEG/hunet2-hpca_t_ft
    # kvasirSEG/hunet-swta
    # kvasirSEG/hunet-swta_ft
    # kvasirSEG/hunet-swta_t
    # kvasirSEG/hunet-swta_t_ft
    kvasirSEG/hfcn32s-swta
    kvasirSEG/hfcn32s-swta_ft
    kvasirSEG/hfcn32s-swta_t
    kvasirSEG/hfcn32s-swta_t_ft
    #kvasirSEG/hunet2-swta_ft
    #kvasirSEG/hunet2-hpca_ft
    #################################
    # DataScienceBowl2018 Dataset
    #################################
    # datasciencebowl2018/hunet-hpca
    # datasciencebowl2018/hunet-hpca_ft
    # datasciencebowl2018/hunet-hpca_t
    # datasciencebowl2018/hunet-hpca_t_ft
    datasciencebowl2018/hfcn32s-hpca
    datasciencebowl2018/hfcn32s-hpca_ft
    datasciencebowl2018/hfcn32s-hpca_t
    datasciencebowl2018/hfcn32s-hpca_t_ft
    #datasciencebowl2018/hunet2-hpca_ft
    #datasciencebowl2018/hunet2-hpca_t_ft
    # datasciencebowl2018/hunet-swta
    # datasciencebowl2018/hunet-swta_ft
    # datasciencebowl2018/hunet-swta_t
    # datasciencebowl2018/hunet-swta_t_ft
    datasciencebowl2018/hfcn32s-swta
    datasciencebowl2018/hfcn32s-swta_ft
    datasciencebowl2018/hfcn32s-swta_t
    datasciencebowl2018/hfcn32s-swta_t_ft
    #datasciencebowl2018/hunet2-swta_ft
    #datasciencebowl2018/hunet2-hpca_ft
)

# Train & Evaluate (k-cross validation)
for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do    # Multiple repetitions of the same experiment, varying the validation bucket
    for EXP in ${EXPS[@]}; do
        case $EXP in
            */hunet-swta*)
                case $EXP in
                    ph2*)
                        CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.cross_val_bucket_validation_index=$REP model.hebb.k=$INV_TEMP_PH2;;
                    isic2016*)
                        CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.cross_val_bucket_validation_index=$REP model.hebb.k=$INV_TEMP_ISIC2016;;
                    kvasirSEG*)
                        CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.cross_val_bucket_validation_index=$REP model.hebb.k=$INV_TEMP_KvasirSEG;;
                    datasciencebowl2018*)
                        CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.cross_val_bucket_validation_index=$REP model.hebb.k=$INV_TEMP_DataScienceBowl2018;;
                esac;;
            *)
                CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python train.py experiment=$EXP data.train.cross_val_bucket_validation_index=$REP;;
        esac
    done
done


# Test 
for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
    for EXP in ${EXPS[@]}; do
        case $EXP in 
            ph2*)
                case $EXP in
                    */hunet-swta*)
                        CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-$INV_TEMP_PH2/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/PH2 --in-memory True;;
                    *)
                        CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-1/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/PH2 --in-memory True;;
                esac;;
            isic2016*)
                case $EXP in
                    */hunet-swta*)
                        CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-$INV_TEMP_ISIC2016/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/ISIC2016 --in-memory True;;
                    *)
                        CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-1/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/ISIC2016 --in-memory True;;
                esac;;
            kvasirSEG*)
                case $EXP in
                    */hunet-swta*)
                        CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-$INV_TEMP_KvasirSEG/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/KvasirSEG --in-memory True;;
                    *)
                        CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-1/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/KvasirSEG --in-memory True;;
                esac;;
            datasciencebowl2018*)
                case $EXP in
                    */hunet-swta*)
                        CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-$INV_TEMP_DataScienceBowl2018/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/DataScienceBowl2018 --in-memory True;;
                    *)
                        CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py $EVAL_EXP_ROOT/experiment=$EXP/inv_temp-1/regime-1.0/run-$REP --data-root $EVAL_DATA_ROOT/DataScienceBowl2018 --in-memory True;;
                esac;;
        esac
    done
done
