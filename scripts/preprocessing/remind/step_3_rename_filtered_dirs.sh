#!/bin/bash

# Base path for the dataset
base_path="/home/adil/Documents/TUE/preparationPhase/myProject/data/ReMIND_dataset/ReMIND-Manifest-Sept-2023/ReMIND"

# Loop through each patient directory
for patient_dir in "$base_path"/ReMIND-*; do
    # Check if it is a directory
    if [ -d "$patient_dir" ]; then
        # Rename Preop directories
        for preop_dir in "$patient_dir"/*Preop*; do
            if [ -d "$preop_dir" ]; then
                # Extract the new name by stripping the unwanted part
                new_name="${preop_dir%%Preop*}Preop"
                mv "$preop_dir" "$new_name"
                echo "Renamed directory: $preop_dir to $new_name"
            fi
        done
        
        # Rename Intraop directories
        for intraop_dir in "$patient_dir"/*Intraop*; do
            if [ -d "$intraop_dir" ]; then
                # Extract the new name by stripping the unwanted part
                new_name="${intraop_dir%%Intraop*}Intraop"
                mv "$intraop_dir" "$new_name"
                echo "Renamed directory: $intraop_dir to $new_name"
            fi
        done
    fi
done
