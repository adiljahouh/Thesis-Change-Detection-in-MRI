#!/bin/bash

# Base directory for ReMIND_unused dataset
BASE_DIR=~/Documents/TUE/Thesis\ +\ Prep\ Phase/myProject/data/ReMIND_dataset/ReMIND-Manifest-Sept-2023/ReMIND/ReMIND-Unused/

# Keywords to search for
KEYWORD_T1="3DAXT1postcontrast"
KEYWORD_T1_ALT="3DAXT1precontrast"
KEYWORD_TUMOR_SEG="tumor seg"

counter=0

# Iterate through each patient directory
for patient_dir in "$BASE_DIR"/*; do
    if [ -d "$patient_dir" ]; then
        echo "Checking patient: $(basename "$patient_dir")"
        
        # Search for Preop and Intraop directories
        preop_dir=$(find "$patient_dir" -type d -name "*Preop*" | head -n 1)
        intraop_dir=$(find "$patient_dir" -type d -name "*Intraop*" | head -n 1)
        
        if [ -d "$preop_dir" ] && [ -d "$intraop_dir" ]; then

            # First try finding `3DAXT1postcontrast`, excluding seg folders
            preop_match=$(find "$preop_dir" -type d -not -path "*seg*" -name "*$KEYWORD_T1*")
            intraop_match=$(find "$intraop_dir" -type d -not -path "*seg*" -name "*$KEYWORD_T1*")
            
            # If `3DAXT1postcontrast` not found, fall back to `3DAXT1`, still excluding seg folders
            if [ -z "$preop_match" ]; then
                preop_match=$(find "$preop_dir" -type d -not -path "*seg*" -name "*$KEYWORD_T1_ALT*")
            fi
            if [ -z "$intraop_match" ]; then
                intraop_match=$(find "$intraop_dir" -type d -not -path "*seg*" -name "*$KEYWORD_T1_ALT*")
            fi

            if [ -n "$preop_match" ] && [ -n "$intraop_match" ]; then            
                echo "T1 scan found for $(basename "$patient_dir")"

                counter=$((counter+1))

                # Check and create T1_converted directories if they do not exist
                [ ! -d "$preop_dir/T1_converted" ] && mkdir -p "$preop_dir/T1_converted"
                [ ! -d "$intraop_dir/T1_converted" ] && mkdir -p "$intraop_dir/T1_converted"

                # Check if a `.nii.gz` file with 'postcontrast' exists in the directory
                preop_nifti_exists=$(find "$preop_dir/T1_converted" -type f -iname "*.nii.gz" | wc -l)
                intraop_nifti_exists=$(find "$intraop_dir/T1_converted" -type f -iname "*.nii.gz" | wc -l)

                if [ "$preop_nifti_exists" -gt 0 ]; then
                    echo "Skipping conversion for $(basename "$preop_dir") (preop, already converted)"
                else
                    echo "Converting $preop_match to T1_converted..."
                    dcm2niix -z y -o "$preop_dir/T1_converted" "$preop_match"
                fi

                if [ "$intraop_nifti_exists" -gt 0 ]; then
                    echo "Skipping conversion for $(basename "$intraop_dir") (intraop, already converted)"
                else
                    echo "Converting $intraop_match to T1_converted..."
                    dcm2niix -z y -o "$intraop_dir/T1_converted" "$intraop_match"
                fi
            fi

            # Convert tumor segmentation (no need to check for T1 keywords)
            for seg_dir in "$preop_dir" "$intraop_dir"; do
                tumor_seg_match=$(find "$seg_dir" -type d -name "*$KEYWORD_TUMOR_SEG*")
                
                if [ -n "$tumor_seg_match" ]; then
                    echo "Tumor segmentation found in $(basename "$seg_dir")"

                    tumor_output_dir="$seg_dir/tumor_converted"
                    [ ! -d "$tumor_output_dir" ] && mkdir -p "$tumor_output_dir"

                    tumor_nifti="$tumor_output_dir/1.nii.gz"

                    if [ ! -f "$tumor_nifti" ]; then
                        dicom_file=$(find "$tumor_seg_match" -type f -iname "*.dcm" | head -n 1)
                        if [ -n "$dicom_file" ]; then
                            echo "Converting tumor segmentation in $(basename "$seg_dir")..."
                            segimage2itkimage --outputDirectory "$tumor_output_dir" --inputDICOM "$dicom_file" -t nii
                        else
                            echo "No valid DICOM file found in tumor segmentation directory, skipping."
                        fi
                    else
                        echo "Tumor segmentation already converted, skipping."
                    fi
                fi
            done

        else
            echo "Preop or Intraop directory not found for $(basename "$patient_dir")"
        fi
    else
        echo "Skipping: $(basename "$patient_dir") (Not a directory)"
    fi
done

echo "Total patients with T1 scan converted: $counter"