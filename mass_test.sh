#!/bin/bash

# Define ranges for margin and threshold
for margin in $(seq 1.0 1.0 7.0 | sed 's/,/./g'); do
    for threshold in $(seq 0.1 0.1 1.0 | sed 's/,/./g'); do
        echo "Running with margin=$margin and threshold=$threshold"
        python -u src/main.py \
            --model MLO \
            --skip 100 \
            --batch_size 16 \
            --loss TCL \
            --margin "$margin" \
            --threshold "$threshold" \
            --dist_flag l2 \
            --patience 1 \
            --load_slices
    done
done
