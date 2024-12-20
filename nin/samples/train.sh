#!/bin/bash

# Build Path

cd "$(dirname "$0")"
BASE_PATH=$(../../check_path.sh)
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="../nin/train.py"
RES_PATH="${BASE_PATH}/model/nin/fashionmnist"
NAME=$(date +'%Y_%m_%d_%H_%M')
mkdir -p "${RES_PATH}/${NAME}"

python $PYTHON_SCRIPT \
    --comment "NIN on Fashion with ReduceLROnPlateau on SGD" \
    --model_base_path "${RES_PATH}/${NAME}" \
    --dataset_path "${BASE_PATH}/torchvision" \
    --image_size 224 \
    --num_classes 10 \
    --seed 42 \
    --batch_size 32 \
    --lr 0.001 \
    --max_epoch 100 \
    --max_reduce 3 \
    --factor 0.1 \
    --train_mode "Holdout" \
    --val 0.2