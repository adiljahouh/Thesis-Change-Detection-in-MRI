#!/bin/bash

# Base directory where the ReMIND dataset is stored
BASE_DIR=~/Documents/TUE/preparationPhase/myProject/data/ReMIND_dataset/ReMIND-Manifest-Sept-2023/ReMIND

# Iterate through each Unused_* directory
for unused_dir in "$BASE_DIR"/Unused_*; do
    if [ -d "$unused_dir" ]; then
        # Get the new directory name by removing the "Unused_" prefix
        new_name="${unused_dir/Unused_/}"
        echo "Renaming $unused_dir to $new_name"
        mv "$unused_dir" "$new_name"
    else
        echo "Skipping: $unused_dir (Not a directory)"
    fi
done

echo "Renaming complete."