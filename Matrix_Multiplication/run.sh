#!/bin/bash
nvcc main_1D.cu
set -e
echo "row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)" > res.csv

base=1024
for i in {1..4}
do
    ./a.out $base $base $base 1024 #| tail - n 1 >> res.csv
    base=$((2 * $base))
done

col_B=1024
for i in {1..3}
do
    ./a.out 1024 1024 $col_B 1024 #| tail - n 1 >> res.csv
    col_B=$((2 * $col_B))
done

thread_cnt=16
for i in {1..4}
do
    ./a.out 1024 1024 1024 $thread_cnt #| tail -n 1 >> res.csv
    thread_cnt=$((4 * $thread_cnt))
done
