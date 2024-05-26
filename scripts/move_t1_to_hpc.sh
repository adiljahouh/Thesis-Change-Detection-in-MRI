for file in ../data/processed/*/*/*/t1_ants_aligned.nii.gz; do
    wildcard=$(echo $file | cut -d '/' -f 3,4,5)
    ssh u50@145.38.193.83 "mkdir -p /home/u50/data/mri_data_and_models/processed/$wildcard/"
    scp -rp $file u50@145.38.193.83:/home/u50/data/mri_data_and_models/processed/$wildcard/
done
