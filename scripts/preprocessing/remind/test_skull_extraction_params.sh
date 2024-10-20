#!/bin/bash

# Absolute path of the input NIfTI file
input_file="/home/adil/Documents/TUE/preparationPhase/myProject/data/ReMIND_dataset/ReMIND-Manifest-Sept-2023/ReMIND/ReMIND-103/12-25-1982-Preop-65502/T1_converted/2.000000-3DAXT1postcontrast-38037_3D_AX_T1_postcontrast_19821225152927_2.nii.gz"

# Check if the input file exists
if [[ ! -f "$input_file" ]]; then
  echo "Error: Input file $input_file not found."
  exit 1
fi

# Absolute path for output directory
output_dir="/home/adil/Documents/TUE/preparationPhase/myProject/data/ReMIND_dataset/ReMIND-Manifest-Sept-2023/ReMIND/ReMIND-103/12-25-1982-Preop-65502/T1_converted/bet_outputs"
rm -rf "$output_dir"
mkdir -p "$output_dir"

# Define the ranges for f and g values
f_values=(0.2 0.4 0.5 0.6 0.8 1) # Modify these as needed
g_values=(-0.5 -0.3 -0.1 0 0.1 0.3 0.5)         # Modify these as needed

# Apply bet command with different combinations of f and g values
for f in "${f_values[@]}"; do
  for g in "${g_values[@]}"; do
    output_file="$output_dir/f_${f}_g_${g}.nii.gz"
    echo "Running bet with f=$f and g=$g, outputting to $output_file"
    
    # Apply bet
    bet "$input_file" "$output_file" -f "$f" -g "$g"
    
    # Check if bet was successful
    if [[ $? -ne 0 ]]; then
      echo "Error running bet with f=$f and g=$g."
      exit 1
    fi
  done
done

echo "All bet operations completed."
