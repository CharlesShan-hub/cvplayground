#!/bin/bash

# Get Base Path

cd "$(dirname "$0")"
BASE_PATH=$("../../check_path.sh")
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="../nin/test.py"
RES_PATH="${BASE_PATH}/model/nin/fashionmnist"

python $PYTHON_SCRIPT \
    --comment "Nin on FashionMNIST" \
    --model_path "${RES_PATH}/8702/checkpoints/40.pt" \
    --dataset_path "${BASE_PATH}/torchvision"\
    --image_size 224 \
    --num_classes 10 \
    --batch_size 32