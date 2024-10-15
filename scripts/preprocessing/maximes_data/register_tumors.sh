#!/bin/bash

# Define the base directories
RAW_DIR="/home/adil/Documents/TUE/preparationPhase/myProject/data/raw/preop/BTC-preop/derivatives/tumor_masks"
PROCESSED_DIR="/home/adil/Documents/TUE/preparationPhase/myProject/data/processed/preop/BTC-preop"

# Loop through patients from PAT01 to PAT31
for i in $(seq -w 01 31); do
  patient="PAT${i}"
  echo "Processing ${patient}..."
  
  # Define paths for the current patient
  input_image="${RAW_DIR}/sub-${patient}/anat/sub-${patient}_space_T1_label-tumor.nii"
  reference_image="${PROCESSED_DIR}/sub-${patient}/t1_ants_aligned.nii.gz"
  output_image="${PROCESSED_DIR}/sub-${patient}/aligned_tumor_mask.nii.gz"
  warp_file="${PROCESSED_DIR}/sub-${patient}/ANTS1Warp.nii.gz"
  affine_file="${PROCESSED_DIR}/sub-${patient}/ANTS0GenericAffine.mat"

  # Check if the input image exists before processing
  if [ -f "${input_image}" ]; then
    # Run antsApplyTransforms
    antsApplyTransforms -d 3 -i "${input_image}" -r "${reference_image}" -o "${output_image}" -t "${warp_file}" -t "${affine_file}"
    echo "Completed processing ${patient}"
  else
    echo "Input image ${input_image} for ${patient} does not exist. Skipping."
  fi
done
