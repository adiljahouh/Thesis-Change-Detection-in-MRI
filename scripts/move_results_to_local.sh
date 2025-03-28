for dir in $(ssh u50@hpc_cloud "ls -d ~/results/MLO*/train_test"); do
    base_dir=$(basename $(dirname "$dir"))  # Extracts the MLO_* directory name
    local_dir="/home/adil/Documents/TUE/ThesisPrepPhase/myProject/results/$base_dir"

    mkdir -p "$local_dir"  # Ensure the local directory exists

    # Use rsync to copy only files, excluding subdirectories
    rsync -avz --ignore-existing --exclude=*/ u50@hpc_cloud:$dir/ "$local_dir/"
done
