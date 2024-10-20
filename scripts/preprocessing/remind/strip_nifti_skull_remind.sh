#!/bin/bash

# Base directory where the ReMIND dataset is stored
BASE_DIR=~/Documents/TUE/preparationPhase/myProject/data/ReMIND_dataset/ReMIND-Manifest-Sept-2023/ReMIND

# Iterate through each patient directory
for patient_dir in "$BASE_DIR"/*; do
    if [ -d "$patient_dir" ]; then
        echo "Processing patient: $(basename "$patient_dir")"

        # Search for all T1_converted directories
        find "$patient_dir" -type d -name "*T1_converted*" | while read -r t1_converted_dir; do
            if [ -d "$t1_converted_dir" ]; then
                echo "Found T1_converted directory: $t1_converted_dir"

                # Iterate through all .nii.gz files in the T1_converted directory
                for nii_file in "$t1_converted_dir"/*.nii.gz; do
                    if [ -f "$nii_file" ]; then
                        # Define output file name
                        output_file="$t1_converted_dir/t1_aligned_stripped.nii.gz"
                        
                        # Check if the output file already exists
                        if [ ! -f "$output_file" ]; then
                            echo "Applying BET on $nii_file with f=0.4..."
                            # Run the BET command
                            bet "$nii_file" "$output_file" -f 0.4 -g 0
                        else
                            echo "Skipping $nii_file; output file $output_file already exists."
                        fi
                    else
                        echo "No .nii.gz files found in $t1_converted_dir"
                    fi
                done
            else
                echo "T1_converted directory not found in $(basename "$patient_dir")"
            fi
        done

    else
        echo "Skipping: $(basename "$patient_dir") (Not a directory)"
    fi
done

echo "BET processing complete."
