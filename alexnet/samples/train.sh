#!/bin/bash

# Build Path

cd "$(dirname "$0")"
BASE_PATH=$(../../check_path.sh)
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="../alexnet/train.py"
RES_PATH="${BASE_PATH}/model/alexnet/mnist"
NAME=$(date +'%Y_%m_%d_%H_%M')
mkdir -p "${RES_PATH}/${NAME}"

python $PYTHON_SCRIPT \
    --comment "AlexNet on MNIST with ReduceLROnPlateau on SGD" \
    --model_base_path "${RES_PATH}/${NAME}" \
    --dataset_path "${BASE_PATH}/torchvision" \
    --image_size 224 \
    --num_classes 10 \
    --seed 42 \
    --batch_size 16 \
    --lr 0.03 \
    --max_epoch 100 \
    --max_reduce 3 \
    --factor 0.1 \
    --train_mode "Holdout" \
    --val 0.2