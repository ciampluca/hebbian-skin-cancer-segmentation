#!/bin/bash

# We used only training set since it is the only subset with masks

DATASET_ROOT="./"

if [[ -z "$DATASET_ROOT" ]]; then
    DATASET_ROOT="./"
fi

DRIVE_DIR="${DATASET_ROOT}/DRIVE"
DRIVE_URL="./DRIVE_url.txt"


echo "Downloading, extracting, and preparing: DRIVE dataset"
if [[ ! -e "${DRIVE_DIR}" ]]; then
    while read -r line
    do
      gdown "${line}" -O ${DATASET_ROOT}
    done < ${DRIVE_URL}
    unzip ${DATASET_ROOT}/DRIVE.zip -d ${DRIVE_DIR}
    rm ${DATASET_ROOT}/DRIVE.zip
    unzip ${DRIVE_DIR}/training.zip -d ${DRIVE_DIR} && rm ${DRIVE_DIR}/training.zip
    rm ${DRIVE_DIR}/test.zip
    mv ${DRIVE_DIR}/training/images ${DRIVE_DIR}/images
    mv ${DRIVE_DIR}/training/1st_manual ${DRIVE_DIR}/targets
    rm -rf ${DRIVE_DIR}/training
    python prepare_DRIVE.py
else
    echo "Folder ${DRIVE_DIR} already exists. Exiting..."
fi