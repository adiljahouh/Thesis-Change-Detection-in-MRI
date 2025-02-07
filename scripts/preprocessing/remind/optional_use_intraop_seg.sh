#!/bin/bash

# Base directory where the ReMIND dataset is stored
BASE_DIR=~/Documents/TUE/Thesis\ +\ Prep\ Phase/myProject/data/ReMIND_dataset/ReMIND-Manifest-Sept-2023/ReMIND/ReMIND-Unused/

# Keyword for tumor segmentation in intraoperative scans
KEYWORD_TUMOR="tumorresidual seg"

counter=0

# Iterate through each patient directory
for patient_dir in "$BASE_DIR"/*; do
    if [ -d "$patient_dir" ]; then
        echo "Checking patient: $(basename "$patient_dir")"
        
        # Search for Intraop directory
        intraop_dir=$(find "$patient_dir" -type d -name "*Intraop*" | head -n 1)
        
        if [ -d "$intraop_dir" ]; then

            # Find any tumor segmentation in the intraop directory (remove T1 filter)
            tumor_seg_match=$(find "$intraop_dir" -type d | grep -i "$KEYWORD_TUMOR")

            if [ -n "$tumor_seg_match" ]; then
                echo "Tumor segmentation found for patient: $(basename "$patient_dir")"

                counter=$((counter+1))

                # Define output directory for converted tumor segmentation
                tumor_output_dir="$intraop_dir/tumor_converted"
                [ ! -d "$tumor_output_dir" ] && mkdir -p "$tumor_output_dir"

                # Define output NIfTI file path
                tumor_nifti="$tumor_output_dir/1.nii.gz"

                # Check if tumor segmentation already converted
                if [ ! -f "$tumor_nifti" ]; then
                    echo "Converting tumor segmentation for $(basename "$patient_dir")..."
                    segimage2itkimage --outputDirectory "$tumor_output_dir" --inputDICOM "$tumor_seg_match/1-1.dcm" -t nii
                else
                    echo "Tumor segmentation already converted, skipping."
                fi
            else
                echo "No tumor segmentation found for $(basename "$patient_dir")"
            fi

        else
            echo "Intraop directory not found for $(basename "$patient_dir")"
        fi
    else
        echo "Skipping: $(basename "$patient_dir") (Not a directory)"
    fi
done

echo "Total patients with tumor segmentation converted: $counter"
