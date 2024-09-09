nohup python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 7.0 > output.log &

#commands TCL
# python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 7.0 --loss TCL --threshold 0.3

# commands CL
# python -u src/main.py --model custom --lr 0.001 --epochs 200 --patience 8 --dist_flag l2 --margin 7.0 --loss CL --threshold 0.3