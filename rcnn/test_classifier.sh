#!/bin/bash

# Get Base Path

BASE_PATH=$(../check_path.sh)
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="test_classifier.py"
RES_PATH="${BASE_PATH}/Model/RCNN/Flowers17"

# pth on the Mac PC
python $PYTHON_SCRIPT \
    --comment "(RCNN on 17flowers) step1: Train AlexNet Classifier(with pre-trained model)" \
    --model_path "${RES_PATH}/AlexNet_Classifier/checkpoints/43.pt" \
    --dataset_path "${BASE_PATH}/torchvision"\
    --num_classes 17 \
    --image_size 224 \
    --batch_size 32