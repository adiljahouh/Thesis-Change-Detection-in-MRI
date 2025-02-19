for dir in $(ssh u50@hpc_cloud "ls -d ~/results/MLO*/train_test"); do
    base_dir=$(basename $(dirname "$dir"))  # Gets MLO_l2_lr-... directory name
    mkdir -p "/home/adil/Documents/TUE/ThesisPrepPhase/myProject/results/$base_dir"  # Creates matching directory locally
    scp u50@hpc_cloud:$dir/* "/home/adil/Documents/TUE/ThesisPrepPhase/myProject/results/$base_dir/"
done
