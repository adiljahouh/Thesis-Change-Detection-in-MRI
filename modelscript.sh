

nohup python -u src/main.py --model MLO --skip 3 --batch_size 16 --loss TCL --margin 6 --threshold 0.5 --dist_flag l2 --patience 10 --load_slices > output.log &