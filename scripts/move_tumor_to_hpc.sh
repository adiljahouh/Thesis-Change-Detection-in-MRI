for file in ./data/raw/preop/BTC-preop/derivatives/tumor_masks/*/anat/*; do
    # Extract the sub-PATXX directory structure
    wildcard=$(echo $file | cut -d '/' -f 8,9)

    # Define the remote destination path and create the corresponding directory structure
    ssh hpc_cloud "mkdir -p /home/u50/data/mri_data_and_models/raw/preop/BTC-preop/derivatives/tumor_masks/$wildcard/"
    echo "Created directory: /home/u50/data/mri_data_and_models/raw/preop/BTC-preop/derivatives/tumor_masks/$wildcard/"
    scp -rp $file u50@145.38.193.83:/home/u50/data/mri_data_and_models/raw/preop/BTC-preop/derivatives/tumor_masks/$wildcard/
done
