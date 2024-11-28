#nohup python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 7.0 > output.log &

#commands TCL
# echo "Running CL"
# python -u src/main.py --model deeplab --skip 4 --batch_size 4


##dev 
# python -u src/main.py --model SLO --preop_dir "./data/processed/preop/BTC-preop" --tumor_dir "./data/raw/preop/BTC-preop/derivatives/tumor_masks"  --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 5.0 --loss TCL --threshold 0.3 --skip 2

#python -u src/main.py --model MLO --skip 2 --batch_size 16 --loss TCL --threshold 0.2 --load_slices --patience 10
 
python -u src/predictions.py --model MLO --mode augmented --model_path ./results/PROD_MLO_l2_lr-0.001_marg-5.0_thresh-0.3_loss-TCL/model.pth