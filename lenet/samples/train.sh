#!/bin/bash

# Build Path

cd "$(dirname "$0")"
BASE_PATH=$(../../check_path.sh)
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="../lenet/train.py"
RES_PATH="${BASE_PATH}/model/lenet/mnist"
NAME=$(date +'%Y_%m_%d_%H_%M_%S')
mkdir -p "${RES_PATH}/${NAME}"

python $PYTHON_SCRIPT \
    --comment "LeNet on MNIST with ReduceLROnPlateau on SGD" \
    --model_base_path "${RES_PATH}/${NAME}" \
    --dataset_path "${BASE_PATH}/torchvision" \
    --image_size 28 \
    --num_classes 10 \
    --use_relu False \
    --use_max_pool False \
    --seed 32 \
    --batch_size 16 \
    --lr 0.3 \
    --max_epoch 100 \
    --max_reduce 3 \
    --factor 0.1 \
    --train_mode "Holdout" \
    --val 0.2