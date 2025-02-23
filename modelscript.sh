#nohup python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 7.0 > output.log &

#commands TCL
# echo "Running CL"
# python -u src/main.py --model deeplab --skip 4 --batch_size 4


##dev 
# python -u src/main.py --model SLO --preop_dir "./data/processed/preop/BTC-preop" --tumor_dir "./data/raw/preop/BTC-preop/derivatives/tumor_masks"  --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 5.0 --loss TCL --threshold 0.3 --skip 2


nohup python -u src/main.py --model MLO --skip 2 --batch_size 16 --loss TCL --margin 99.0 --threshold 11 --dist_flag l2 --patience 10 --load_slices > output.log &
# nohup python -u src/main.py --model MLO --skip 1 --batch_size 16 --loss TCL --margin 4.0 --threshold 0.1 --dist_flag l2 --patience 10 --load_slices > output.log &
# nohup python -u src/main.py --model MLO --skip 1 --batch_size 16 --loss TCL --margin 6.0 --threshold 0.1 --dist_flag l2 --patience 10 --load_slices > output.log &
# nohup python -u src/main.py --model MLO --skip 1 --batch_size 16 --loss TCL --margin 1.0 --threshold 0.0 --dist_flag l2 --patience 10 --load_slices > output.log &
