#!/bin/bash

DATASET_ROOT="./"

if [[ -z "$DATASET_ROOT" ]]; then
    DATASET_ROOT="./"
fi

KvasirSEG_DIR="${DATASET_ROOT}/KvasirSEG"
KvasirSEG_URL="./KvasirSEG_url.txt"


echo "Downloading, extracting, and preparing: KvasirSEG dataset"
if [[ ! -e "${KvasirSEG_DIR}" ]]; then
    while read -r line
    do
      gdown "${line}" -O ${DATASET_ROOT}
    done < ${KvasirSEG_URL}
    unzip ${DATASET_ROOT}/kvasir-seg.zip && rm ${DATASET_ROOT}/kvasir-seg.zip
    python prepare_KvasirSEG.py
    rm -rf ${DATASET_ROOT}/Kvasir-SEG
else
    echo "Folder ${KvasirSEG_DIR} already exists. Exiting..."
fi