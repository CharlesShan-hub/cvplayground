#!/bin/bash

# Get Base Path

cd "$(dirname "$0")"
BASE_PATH=$("../../check_path.sh")
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="../deepfuse/test.py"

python $PYTHON_SCRIPT \
    --dataset_path "${BASE_PATH}/torchvision"\
    --save_path "${BASE_PATH}/temp/deepfuse"\
    --pre_trained "/Volumes/Charles/DateSets/Model/DeepFuse/DeepFuse_model.pth"

