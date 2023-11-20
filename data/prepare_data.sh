#!/bin/bash

echo "Creating datasets"

bash prepare_PH2.sh
#bash prepare_BrainMRI.sh
bash prepare_KvasirSEG.sh
#bash prepare_DRIVE.sh
bash prepare_DataScienceBowl2018.sh
bash prepare_ISIC2016.sh
bash prepare_GlaS.sh

echo "Done. Exiting..."
