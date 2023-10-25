#!/bin/bash

# We used only training set since it is the only subset with masks

DATASET_ROOT="./"

if [[ -z "$DATASET_ROOT" ]]; then
    DATASET_ROOT="./"
fi

DataScienceBowl2018_DIR="${DATASET_ROOT}/DataScienceBowl2018"
DataScienceBowl2018_URL="./DataScienceBowl2018_url.txt"


echo "Downloading, extracting, and preparing: DataScienceBowl2018 dataset"
if [[ ! -e "${DataScienceBowl2018_DIR}" ]]; then
    while read -r line
    do
      gdown "${line}" -O ${DATASET_ROOT}
    done < ${DataScienceBowl2018_URL}
    unzip ${DATASET_ROOT}/data-science-bowl-2018.zip -d ${DataScienceBowl2018_DIR}
    rm ${DATASET_ROOT}/data-science-bowl-2018.zip
    mkdir ${DataScienceBowl2018_DIR}/train_data
    unzip ${DataScienceBowl2018_DIR}/stage1_train.zip -d ${DataScienceBowl2018_DIR}/train_data
    rm ${DataScienceBowl2018_DIR}/*.zip
    python prepare_DataScienceBowl2018.py
    rm -rf ${DataScienceBowl2018_DIR}/train_data
else
    echo "Folder ${DataScienceBowl2018_DIR} already exists. Exiting..."
fi