#!/bin/bash

# Build Path

BASE_PATH=$(../check_path.sh)
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi

PYTHON_SCRIPT="train_svm.py"
RES_PATH="${BASE_PATH}/Model/RCNN/Flowers17"
NAME=$(date +'%Y_%m_%d_%H_%M')
mkdir -p "${RES_PATH}/${NAME}"


# Run Script

python $PYTHON_SCRIPT \
    --comment "(RCNN on 2flowers) step4: Train SVM" \
    --model_base_path "${RES_PATH}/${NAME}" \
    --dataset_path "${BASE_PATH}/torchvision" \
    --image_size 224