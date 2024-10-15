
#!/bin/bash

# Base directory where the ReMIND dataset is stored
BASE_DIR=~/Documents/TUE/preparationPhase/myProject/data/ReMIND_dataset/ReMIND-Manifest-Sept-2023/ReMIND

# Keyword to search for the T1 contrast sequence
KEYWORD="3DAXT1postcontrast"

# Iterate through each patient directory
for patient_dir in "$BASE_DIR"/*; do
    if [ -d "$patient_dir" ]; then
        echo "Checking patient: $(basename "$patient_dir")"
        
        # Search for Preop and Intraop directories
        preop_dir=$(find "$patient_dir" -type d -name "*Preop*" | head -n 1)
        intraop_dir=$(find "$patient_dir" -type d -name "*Intraop*" | head -n 1)
        
        if [ -d "$preop_dir" ] && [ -d "$intraop_dir" ]; then
            # echo "  Found Preop directory: $preop_dir"
            # echo "  Found Intraop directory: $intraop_dir"

            # Check if keyword exists in both directories
            preop_match=$(find "$preop_dir" -type d -name "*$KEYWORD*")
            intraop_match=$(find "$intraop_dir" -type d -name "*$KEYWORD*")

            if [ -n "$preop_match" ] && [ -n "$intraop_match" ]; then
                echo "  T1 contrast match found in both Preop and Intraop directories:"
                # echo "    Preop: $preop_match"
                # echo "    Intraop: $intraop_match"
            else
                echo "  No matching T1 contrast sequence found in both directories."
            fi
        else
            echo "  Preop or Intraop directory not found."
        fi
    else
        echo "Skipping: $(basename "$patient_dir") (Not a directory)"
    fi
done
