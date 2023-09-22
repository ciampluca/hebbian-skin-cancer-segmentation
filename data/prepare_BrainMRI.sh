#!/bin/bash

DATASET_ROOT="./"

if [[ -z "$DATASET_ROOT" ]]; then
    DATASET_ROOT="./"
fi

BrainMRI_DIR="${DATASET_ROOT}/BrainMRI"
BrainMRI_URL="./BrainMRI_url.txt"


if [[ ! -e "${BrainMRI_DIR}" ]]; then
    echo "Downloading, extracting, and preparing: BrainMRI dataset"
    while read -r line
    do
      gdown "${line}" -O ${DATASET_ROOT}
    done < ${BrainMRI_URL}
    unzip ${DATASET_ROOT}/BrainMRIDataset.zip && rm ${DATASET_ROOT}/BrainMRIDataset.zip && rm -rf ${DATASET_ROOT}/lgg-mri-segmentation
    python prepare_BrainMRI.py
    rm -rf ${DATASET_ROOT}/kaggle_3m
else
    echo "Folder ${BrainMRI_DIR} already exists. Exiting..."
fi