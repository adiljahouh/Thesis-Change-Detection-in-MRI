#!/bin/bash

# Base directory where the ReMIND dataset is stored
BASE_DIR=~/Documents/TUE/preparationPhase/myProject/data/ReMIND_dataset/ReMIND-Manifest-Sept-2023/ReMIND

# Iterate through each patient directory
for patient_dir in "$BASE_DIR"/*; do
    if [ -d "$patient_dir" ]; then
        echo "Processing patient: $(basename "$patient_dir")"

        # Flag to check if any T1_converted directories are found
        found_t1_converted=false

        # Search for all T1_converted directories
        find "$patient_dir" -type d -name "*T1_converted*" | while read -r t1_converted_dir; do
            if [ -d "$t1_converted_dir" ]; then
                found_t1_converted=true
                echo "Found T1_converted directory: $t1_converted_dir"

                # Define output file name prefix
                output_file_prefix="$t1_converted_dir/t1_aligned_stripped"

                # Remove existing output files
                rm -f "$output_file_prefix".nii.gz
                rm -f "$output_file_prefix"_f*.nii.gz

                for nii_file in "$t1_converted_dir"/*.nii.gz; do
                    if [ -f "$nii_file" ]; then
                        # Skip the output file if it matches the current nii_file
                        if [[ "$nii_file" == "$output_file_prefix"* ]]; then
                            continue
                        fi

                        # Determine the range of -f values based on the file path
                        if [[ "$nii_file" == *"Intraop"* ]]; then
                            f_values=$(seq 0.5 0.1 0.8)
                        elif [[ "$nii_file" == *"Preop"* ]]; then
                            f_values=$(seq 0.4 0.1 0.6)
                        else
                            echo "Unknown type for $nii_file; skipping."
                            continue
                        fi

                        # Apply BET for each -f value in the range
                        for f_value in $f_values; do
                            output_file="${output_file_prefix}_f${f_value}.nii.gz"
                            echo "Applying BET on $nii_file with f=$f_value..."
                            bet "$nii_file" "$output_file" -f "$f_value" -g 0
                        done
                    else
                        echo "No .nii.gz files found in $t1_converted_dir"
                    fi
                done
            else
                echo "T1_converted directory not found in $(basename "$patient_dir")"
            fi
        done

        # Rename the patient_dir if no T1_converted directories were found
        if [ "$found_t1_converted" = false ]; then
            new_name="$BASE_DIR/Unused_$(basename "$patient_dir")"
            echo "Renaming $patient_dir to $new_name"
            mv "$patient_dir" "$new_name"
        fi

    else
        echo "Skipping: $(basename "$patient_dir") (Not a directory)"
    fi
done

echo "BET processing complete."