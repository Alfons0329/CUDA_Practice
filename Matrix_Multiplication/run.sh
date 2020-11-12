#!/bin/bash
set -e
echo "row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)" > res.csv

./a.out 16 16 16 1024 | tail -n 1 >> res.csv
./a.out 128 128 128 1024 | tail -n 1 >> res.csv
./a.out 1024 1024 1024 1024 | tail -n 1 >> res.csv
./a.out 16384 16384 16384 16384 | tail -n 1 >> res.csv

col_B=1024
for i in {1..5}
do
    col_B=$((2 * $col_B))
    ./a.out 1024 1024 $col_B 1024 | tail - n 1 >> res.csv
done

thread_cnt=16
for i in {1..4}
do
    thread_cnt=$((2 * $thread_cnt))
    ./a.out 1024 1024 1024 $thread_cnt | tail -n 1 >> res.csv
done
