#!/bin/bash

# Get Base Path

BASE_PATH=$(../check_path.sh)
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="save_feature.py"
RES_PATH="${BASE_PATH}/Model/RCNN/Flowers17"

# pth on the Mac PC
python $PYTHON_SCRIPT \
    --comment "(RCNN on 2flowers) step3: Save Feature from Finetune AlexNet Classifier" \
    --model_path "${RES_PATH}/AlexNet_Finetune/checkpoints/51.pt" \
    --dataset_path "${BASE_PATH}/torchvision"\
    --num_classes 3 \
    --image_size 224 \
    --batch_size 8