#!/bin/bash

echo "Creating datasets"

bash prepare_PH2.sh
bash prepare_KvasirSEG.sh
bash prepare_GlaS.sh

echo "Done. Exiting..."
