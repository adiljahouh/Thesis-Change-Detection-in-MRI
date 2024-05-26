#!/bin/bash

# Define paths to input and reference directories
postop_dir="../data/raw/postop/BTC-postop"
preop_dir="../data/raw//preop/BTC-preop"

# Iterate through each folder in postop_dir
for postop_subdir in "$postop_dir"/*/; do
    echo "Entering directory: $postop_subdir in $postop_dir"
    # Check if the item is a directory
    if [ -d "$postop_subdir" ]; then
        # Extract subject ID from the folder name
        subject=$(basename "$postop_subdir")
        echo "Processing subject: $subject"
        
        # Define paths to input and reference images
        input_image="$postop_subdir/T1w.nii.gz"
        ref_image="$preop_dir/$subject/T1w.nii.gz"
        
        # Check if the reference image exists
        if [ -f "$ref_image" ]; then
            # Run FLIRT command
            flirt -in "$input_image" -ref "$ref_image" -out "$postop_subdir/T1w_aligned.nii.gz" -omat "$postop_subdir/alignment_matrix"
            
            # Rename the reference image to match the output image
            cp "$ref_image" "$preop_dir/$subject/T1w_aligned.nii.gz"

            # Print message when finished processing subject
            echo "Finished processing subject $subject"
        else
            echo "Reference image not found for subject $subject"
        fi
    fi
done