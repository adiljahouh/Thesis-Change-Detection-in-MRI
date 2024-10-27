#!/bin/bash

# Base directory where the ReMIND dataset is stored
BASE_DIR=~/Documents/TUE/preparationPhase/myProject/data/ReMIND_dataset/ReMIND-Manifest-Sept-2023/ReMIND

# Keywords to search for
KEYWORD_T1="3DAXT1postcontrast"
KEYWORD_TUMOR_SEG="tumor seg"
KEYWORD_REF="ref"
counter=0

# Iterate through each patient directory
for patient_dir in "$BASE_DIR"/*; do
    if [ -d "$patient_dir" ]; then
        echo "Checking patient: $(basename "$patient_dir")"
        
        # Search for Preop and Intraop directories
        preop_dir=$(find "$patient_dir" -type d -name "*Preop*" | head -n 1)
        intraop_dir=$(find "$patient_dir" -type d -name "*Intraop*" | head -n 1)
        
        if [ -d "$preop_dir" ] && [ -d "$intraop_dir" ]; then
            # Check if both preop and intraop contain T1 contrast images and skip tumor/ref images
            preop_match=$(find "$preop_dir" -type d -name "*$KEYWORD_T1*" ! -name "*$KEYWORD_TUMOR_SEG*" ! -name "*$KEYWORD_REF*")
            intraop_match=$(find "$intraop_dir" -type d -name "*$KEYWORD_T1*" ! -name "*$KEYWORD_TUMOR_SEG*" ! -name "*$KEYWORD_REF*")

            if [ -n "$preop_match" ] && [ -n "$intraop_match" ]; then
                # Check for the existence of T1_converted directories
                if [ -d "$preop_dir/T1_converted" ] && [ -d "$intraop_dir/T1_converted" ]; then
                    echo "Patient $(basename "$patient_dir") has T1_converted directories."
                    counter=$((counter+1))
                fi
            fi
        else
            echo "Preop or Intraop directory not found."
        fi
    else
        echo "Skipping: $(basename "$patient_dir") (Not a directory)"
    fi
done

echo "Total patients with T1_converted directories: $counter"
