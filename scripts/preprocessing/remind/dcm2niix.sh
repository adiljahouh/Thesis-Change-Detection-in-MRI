#!/bin/bash

# Base directory where the ReMIND dataset is stored
BASE_DIR=~/Documents/TUE/preparationPhase/myProject/data/ReMIND_dataset/ReMIND-Manifest-Sept-2023/ReMIND

# Keywords to search for
KEYWORD_T1="3DAXT1postcontrast"
KEYWORD_TUMOR_SEG="tumor seg"
KEYWORD_REF="ref"
counter= 0
# Iterate through each patient directory
for patient_dir in "$BASE_DIR"/*; do
    if [ -d "$patient_dir" ]; then
        echo "Checking patient: $(basename "$patient_dir")"
        
        # Search for Preop and Intraop directories
        preop_dir=$(find "$patient_dir" -type d -name "*Preop*" | head -n 1)
        intraop_dir=$(find "$patient_dir" -type d -name "*Intraop*" | head -n 1)
        
        if [ -d "$preop_dir" ] && [ -d "$intraop_dir" ]; then

            # Check if T1 contrast keyword exists in both directories and it is not a tumor segmentation or reference image
            preop_match=$(find "$preop_dir" -type d -name "*$KEYWORD_T1*" ! -name "*$KEYWORD_TUMOR_SEG*" ! -name "*$KEYWORD_REF*")
            intraop_match=$(find "$intraop_dir" -type d -name "*$KEYWORD_T1*" ! -name "*$KEYWORD_TUMOR_SEG*" ! -name "*$KEYWORD_REF*")
            if [ -n "$preop_match" ] && [ -n "$intraop_match" ]; then            

                # Check for tumor segmentation, reference, and T1 contrast in Preop
                tumor_seg_match=$(find "$preop_dir" -type d | grep -i "$KEYWORD_TUMOR_SEG" | grep -i "$KEYWORD_REF" | grep -i "$KEYWORD_T1")

                if [ -n "$tumor_seg_match" ]; then
                    # echo "all matches found for $(basename "$patient_dir")"
                    # echo "preop_match: $preop_match"
                    # echo "intraop_match: $intraop_match"
                    # echo "tumor_seg_match: $tumor_seg_match"
                    
                    counter=$((counter+1))
                    echo cleaning directories...
                    rm -rf "$preop_dir/T1_converted"
                    rm -rf "$intraop_dir/T1_converted"
                    rm -rf "$preop_dir/tumor_converted"
                    mkdir -p "$preop_dir/T1_converted"
                    mkdir -p "$intraop_dir/T1_converted"
                    mkdir -p "$preop_dir/tumor_converted"
                    echo converting to NiFTI...
                    
                    dcm2niix -z y  -o "$preop_dir/T1_converted" "$preop_match"
                    dcm2niix -z y  -o "$intraop_dir/T1_converted" "$intraop_match"
                    segimage2itkimage --outputDirectory "$preop_dir/tumor_converted" --inputDICOM "$tumor_seg_match/1-1.dcm" -t  nii

                fi
            fi

        else
            echo " Preop or Intraop directory not found."
        fi
    else
        echo "Skipping: $(basename "$patient_dir") (Not a directory)"
    fi
done
echo "Total cancer patients is $counter"