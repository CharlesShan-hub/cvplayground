#!/bin/bash

# Get Base Path

cd "$(dirname "$0")"
BASE_PATH=$("../../check_path.sh")
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="../yolov1/dataset.py"

python $PYTHON_SCRIPT \
    --dataset_path "${BASE_PATH}/torchvision"\
    --width 448 \
    --height 448 \
    --S 7 \
    --B 2 \
    --C 20 \
    --normalize False \
    --augment False
