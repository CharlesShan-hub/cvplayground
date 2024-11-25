#!/bin/bash

# Get Base Path

cd "$(dirname "$0")"
BASE_PATH=$("../../check_path.sh")
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="../lenet/test.py"
RES_PATH="${BASE_PATH}/Model/LeNet/MNIST"

python $PYTHON_SCRIPT \
    --comment "LeNET on MNNIST" \
    --model_path "${RES_PATH}/2024_11_01_16_39/checkpoints/13.pt" \
    --dataset_path "${BASE_PATH}/torchvision"\
    --image_size 28 \
    --num_classes 10 \
    --use_relu False \
    --use_max_pool False \
    --batch_size 32
