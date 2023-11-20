#!/bin/bash

DATASET_ROOT="./"

if [[ -z "$DATASET_ROOT" ]]; then
    DATASET_ROOT="./"
fi

GlaS_DIR="${DATASET_ROOT}/GlaS"


echo "Downloading, extracting, and preparing: GlaS dataset"
if [[ ! -e "${GlaS_DIR}" ]]; then
    kaggle datasets download -d sani84/glasmiccai2015-gland-segmentation
    unzip ${DATASET_ROOT}/glasmiccai2015-gland-segmentation.zip && rm ${DATASET_ROOT}/glasmiccai2015-gland-segmentation.zip
    python prepare_GlaS.py
    rm -rf ${DATASET_ROOT}/Warwick_QU_Dataset
else
    echo "Folder ${GlaS_DIR} already exists. Exiting..."
fi