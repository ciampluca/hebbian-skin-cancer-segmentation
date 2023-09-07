#!/bin/bash

DATASET_ROOT="./"

if [[ -z "$DATASET_ROOT" ]]; then
    DATASET_ROOT="./"
fi

PH2_DIR="${DATASET_ROOT}/PH2"

PH2_URL="./dataset_url.txt"


# PH2 DATASET
if [[ ! -e "${PH2_DIR}" ]]; then
    echo "Downloading, extracting, and preparing: PH2 dataset"
    read -r line < ${PH2_URL}
    gdown "${line}" -O ${DATASET_ROOT}
    unrar x ${DATASET_ROOT}/PH2Dataset.rar && rm ${DATASET_ROOT}/PH2Dataset.rar
    python prepare_PH2.py
    rm -rf ${DATASET_ROOT}/PH2Dataset
fi