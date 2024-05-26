for btc_dir in ../data/raw/postop/BTC-postop/*/; do
    # Check if ses-postop/anat directory exists
    if [ -d "${btc_dir}ses-postop/anat/" ]; then
        echo "Entering directory: ${btc_dir}ses-postop/anat/"
        # Navigate to ses-postop/anat directory
        cd "${btc_dir}ses-postop/anat/" || exit
        # Execute bet command on each nii.gz file
        for file in *.nii.gz; do
            echo "Running bet on: $file"
            # Extract filename without extension
            filename=$(basename -- "$file")
            filename_no_ext="${filename%.*}"
            # Run bet command and append _mask to output filename
            output_filename="${filename_no_ext}_mask.nii.gz"
            bet "$file" "$output_filename"
        done
        # Return to the original directory
        cd - || exit
    fi
done