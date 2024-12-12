#!/bin/bash

# Define dataset path list
paths=(
    '/root/autodl-fs/data/vision'
    '/home/vision/users/sht/data/vision'
    '/Volumes/Charles/data/vision'
    '/Users/kimshan/Public/data/vision'
)

# Find the first exist path
for path in "${paths[@]}"; do
    if [ -d "$path" ]; then
        echo "$path"
        exit 0
    fi
done
exit 1