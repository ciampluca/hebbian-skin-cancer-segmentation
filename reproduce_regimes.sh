#!/bin/bash

# Concerning swta approaches, this script should be runned after the reproduce_k_search, after setting best temperature value

set -e

REPS=5
START_REP=0
GPU=0
INV_TEMP=1        # to be set accordingly
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
    ph2/hunet-hpca
    ph2/hunet-hpca_ft
    ph2/hunet-hpca_t
    ph2/hunet-hpca_t_ft
    #ph2/hunet2-hpca_ft
    #ph2/hunet2-hpca_t_ft
    ph2/hunet-swta_ft
    ph2/hunet-swta_t_ft
    #ph2/hunet2-swta_ft
    #ph2/hunet2-hpca_ft
    #################################
    # ISIC2016 Dataset
    #################################
    isic2016/hunet-hpca
    isic2016/hunet-hpca_ft
    isic2016/hunet-hpca_t
    isic2016/hunet-hpca_t_ft
    #isic2016/hunet2-hpca_ft
    #isic2016/hunet2-hpca_t_ft
    isic2016/hunet-swta_ft
    isic2016/hunet-swta_t_ft
    #isic2016/hunet2-swta_ft
    #isic2016/hunet2-hpca_ft
    #################################
    # KvasirSEG Dataset
    #################################
    kvasirSEG/hunet-hpca
    kvasirSEG/hunet-hpca_ft
    kvasirSEG/hunet-hpca_t
    kvasirSEG/hunet-hpca_t_ft
    #kvasirSEG/hunet2-hpca_ft
    #kvasirSEG/hunet2-hpca_t_ft
    kvasirSEG/hunet-swta_ft
    kvasirSEG/hunet-swta_t_ft
    #kvasirSEG/hunet2-swta_ft
    #kvasirSEG/hunet2-hpca_ft
    #################################
    # DataScienceBowl2018 Dataset
    #################################
    datasciencebowl2018/hunet-hpca
    datasciencebowl2018/hunet-hpca_ft
    datasciencebowl2018/hunet-hpca_t
    datasciencebowl2018/hunet-hpca_t_ft
    #datasciencebowl2018/hunet2-hpca_ft
    #datasciencebowl2018/hunet2-hpca_t_ft
    datasciencebowl2018/hunet-swta_ft
    datasciencebowl2018/hunet-swta_t_ft
    #datasciencebowl2018/hunet2-swta_ft
    #datasciencebowl2018/hunet2-hpca_ft
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #################################
    # BrainMRI Dataset
    #################################
    # brainMRI/hunet-hpca
    # brainMRI/hunet-hpca_ft
    # brainMRI/hunet-hpca_t
    # brainMRI/hunet-hpca_t_ft
    # #brainMRI/hunet2-hpca_ft
    # #brainMRI/hunet2-hpca_t_ft
    # brainMRI/hunet-swta_ft
    # brainMRI/hunet-swta_t_ft
    # #brainMRI/hunet2-swta_ft
    # #brainMRI/hunet2-hpca_ft
    #################################
    # DRIVE Dataset
    #################################
    # drive/hunet-hpca
    # drive/hunet-hpca_ft
    # drive/hunet-hpca_t
    # drive/hunet-hpca_t_ft
    # #drive/hunet2-hpca_ft
    # #drive/hunet2-hpca_t_ft
    # drive/hunet-swta_ft
    # drive/hunet-swta_t_ft
    # #drive/hunet2-swta_ft
    # #drive/hunet2-hpca_ft
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
for R in ${REGIMES[@]}; do
    for REP in $(seq $(( $START_REP )) $(( $REPS - 1 ))); do
        for EXP in ${EXPS[@]}; do
            case $EXP in 
                ph2*)
                    CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/inv_temp-$INV_TEMP/regime-$R/run-$REP --data-root data/PH2 --in-memory True
                    ;;
                isic2016*)
                    CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/inv_temp-$INV_TEMP/regime-$R/run-$REP --data-root data/ISIC2016 --in-memory True
                    ;;
                kvasirSEG*)
                    CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/inv_temp-$INV_TEMP/regime-$R/run-$REP --data-root data/KvasirSEG --in-memory True
                    ;;
                datasciencebowl2018*)
                    CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/inv_temp-$INV_TEMP/regime-$R/run-$REP --data-root data/DataScienceBowl2018 --in-memory True
                    ;;
                # brainMRI*)
                #     CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/inv_temp-$INV_TEMP/regime-$R/run-$REP --data-root data/BrainMRI --in-memory False
                #     ;;
                # drive*)
                #     CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/inv_temp-$INV_TEMP/regime-$R/run-$REP --data-root data/DRIVE --in-memory True
                #     ;;
            esac
        done
    done
done

