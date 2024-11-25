#!/bin/bash

# Define dataset path list
paths=(
    '/Users/kimshan/Public/data/vision'
    '/root/autodl-fs/DataSets'
    '/Volumes/Charles/DateSets'
    '/home/vision/sht/DataSets'
)

# Find the first exist path
for path in "${paths[@]}"; do
    if [ -d "$path" ]; then
        echo "$path"
        exit 0
    fi
done
exit 1