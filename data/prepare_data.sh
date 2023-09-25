#!/bin/bash

echo "Creating datasets"

bash prepare_PH2.sh
bash prepare_BrainMRI.sh

echo "Done. Exiting..."
