#!/bin/bash

# Get Base Path

cd "$(dirname "$0")"
BASE_PATH=$("../../check_path.sh")
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="../vgg/test.py"
RES_PATH="${BASE_PATH}/model/vgg/imagenet-1k"

python $PYTHON_SCRIPT \
    --comment "Pretrained VGG on ImageNet" \
    --model_path "${RES_PATH}" \
    --dataset_path "${BASE_PATH}/torchvision" \
    --model_name "vgg11" \
    --batch_size 16
