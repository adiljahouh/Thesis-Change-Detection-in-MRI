#nohup python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 7.0 > output.log &

#commands TCL
# echo "Running CL"
python -u src/main.py --model deeplab --skip 4 --batch_size 4
# python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 7.0 --loss CL --threshold 0.0
# python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 9.0 --loss CL --threshold 0.0
# python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 12.0 --loss CL --threshold 0.0

# commands TCL
# echo "Running TCL"
# python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 5.0 --loss TCL --threshold 0.05
# python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 5.0 --loss TCL --threshold 0.1
# python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 5.0 --loss TCL --threshold 1.0

# python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 7.0 --loss TCL --threshold 0.05
# python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 7.0 --loss TCL --threshold 0.1
# python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 7.0 --loss TCL --threshold 0.3
# python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 7.0 --loss TCL --threshold 1.0

# python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 9.0 --loss TCL --threshold 0.05
# python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 9.0 --loss TCL --threshold 0.1
# python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 9.0 --loss TCL --threshold 0.3
# python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 9.0 --loss TCL --threshold 1.0