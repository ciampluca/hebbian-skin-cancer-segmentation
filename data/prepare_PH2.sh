#!/bin/bash

DATASET_ROOT="./"

if [[ -z "$DATASET_ROOT" ]]; then
    DATASET_ROOT="./"
fi

PH2_DIR="${DATASET_ROOT}/PH2"
PH2_URL="./PH2_url.txt"


echo "Downloading, extracting, and preparing: PH2 dataset"
if [[ ! -e "${PH2_DIR}" ]]; then
    while read -r line
    do
      gdown "${line}" -O ${DATASET_ROOT}
    done < ${PH2_URL}
    unrar x ${DATASET_ROOT}/PH2Dataset.rar && rm ${DATASET_ROOT}/PH2Dataset.rar
    python prepare_PH2.py
    rm -rf ${DATASET_ROOT}/PH2Dataset
else
    echo "Folder ${PH2_DIR} already exists. Exiting..."
fi