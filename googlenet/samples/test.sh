#!/bin/bash

# Get Base Path

cd "$(dirname "$0")"
BASE_PATH=$("../../check_path.sh")
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="../googlenet/test.py"
RES_PATH="${BASE_PATH}/model/googlenet/imagenet-1k"

python $PYTHON_SCRIPT \
    --comment "Pretrained GoogLeNet on ImageNet" \
    --model_path "${RES_PATH}" \
    --dataset_path "${BASE_PATH}/torchvision" \
    --batch_size 16
