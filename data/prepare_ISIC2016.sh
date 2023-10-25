#!/bin/bash

DATASET_ROOT="./"

if [[ -z "$DATASET_ROOT" ]]; then
    DATASET_ROOT="./"
fi

ISIC2016_DIR="${DATASET_ROOT}/ISIC2016"
ISIC2016_URL="./ISIC2016_url.txt"


echo "Downloading, extracting, and preparing: ISIC2016 dataset"
if [[ ! -e "${ISIC2016_DIR}" ]]; then
    while read -r line
    do
      gdown "${line}" -O ${DATASET_ROOT}
    done < ${ISIC2016_URL}
    unzip ${DATASET_ROOT}/ISIC_2016.zip -d ${ISIC2016_DIR}
    rm ${DATASET_ROOT}/ISIC_2016.zip
    mv ${ISIC2016_DIR}/ISIC_2016/*.zip ${ISIC2016_DIR} && rm -rf ${ISIC2016_DIR}/ISIC_2016
    unzip ${ISIC2016_DIR}/ISBI2016_ISIC_Part1_Training_Data.zip -d ${ISIC2016_DIR} && unzip ${ISIC2016_DIR}/ISBI2016_ISIC_Part1_Training_GroundTruth.zip -d ${ISIC2016_DIR}
    rm ${ISIC2016_DIR}/ISBI2016_ISIC_Part1_Training_Data.zip && rm ${ISIC2016_DIR}/ISBI2016_ISIC_Part1_Training_GroundTruth.zip
    mv ${ISIC2016_DIR}/ISBI2016_ISIC_Part1_Training_Data ${ISIC2016_DIR}/images
    mv ${ISIC2016_DIR}/ISBI2016_ISIC_Part1_Training_GroundTruth ${ISIC2016_DIR}/targets
    python prepare_ISIC2016.py
else
    echo "Folder ${ISIC2016_DIR} already exists. Exiting..."
fi