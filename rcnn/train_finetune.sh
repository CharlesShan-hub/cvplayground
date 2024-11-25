#!/bin/bash

# Build Path

BASE_PATH=$(../check_path.sh)
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi

PYTHON_SCRIPT="train_finetune.py"
RES_PATH="${BASE_PATH}/Model/RCNN/Flowers17"
NAME=$(date +'%Y_%m_%d_%H_%M')
mkdir -p "${RES_PATH}/${NAME}"


# Run Script

python $PYTHON_SCRIPT \
    --comment "(RCNN on 2flowers) step2: Finetune AlexNet Classifier(with pre-trained model)" \
    --model_base_path "${RES_PATH}/${NAME}" \
    --dataset_path "${BASE_PATH}/torchvision"\
    --pre_trained True \
    --pre_trained_url "${RES_PATH}/AlexNet_Classifier/checkpoints/43.pt" \
    --num_classes 3 \
    --image_size 224 \
    --seed 42 \
    --batch_size 16 \
    --lr 0.3 \
    --max_epoch 100 \
    --max_reduce 3 \
    --factor 0.1 \
    --cooldown 15 \
    --train_mode "Holdout" \
    --val_size 0.1 \
    --test_size 0.1 \
    --seed 42