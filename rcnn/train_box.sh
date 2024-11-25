#!/bin/bash

# Build Path

BASE_PATH=$(../check_path.sh)
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi

PYTHON_SCRIPT="train_box.py"
RES_PATH="${BASE_PATH}/Model/RCNN/Flowers17"
NAME=$(date +'%Y_%m_%d_%H_%M')
mkdir -p "${RES_PATH}/${NAME}"


# Run Script

python $PYTHON_SCRIPT \
    --comment "(RCNN on 2flowers) step5: Train Box" \
    --model_base_path "${RES_PATH}/${NAME}" \
    --dataset_path "${BASE_PATH}/torchvision"\
    --num_classes 3 \
    --image_size 224 \
    --seed 42 \
    --batch_size 16 \
    --lr 0.1 \
    --max_epoch 100 \
    --max_reduce 3 \
    --factor 0.1 \
    --cooldown 5 \
    --train_mode "Holdout" \
    --val_size 0.1 \
    --test_size 0.1 \
    --seed 42