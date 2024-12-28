#!/bin/bash
# this script is used to download the notebook from the github repository and open it in Jupyter Lab on onyxia: it is adapted from: https://github.com/linogaliana/python-datascientist/tree/main

SESSION=$1

WORK_DIR="/home/onyxia/work"
PATH_WITHIN="notebooks"

BASE_URL="https://raw.githubusercontent.com/strayMat/causal-ml-course/main"
NOTEBOOK_PATH="${PATH_WITHIN}/${SESSION}.ipynb"
DOWNLOAD_URL="${BASE_URL}/${NOTEBOOK_PATH}"

# Download the notebook directly using curl
echo $DOWNLOAD_URL
curl -L $DOWNLOAD_URL -o "${WORK_DIR}/${SESSION}.ipynb"

# Open the relevant notebook when starting Jupyter Lab
echo "c.LabApp.default_url = '/lab/tree/${SESSION}.ipynb'" >> /home/onyxia/.jupyter/jupyter_server_config.py