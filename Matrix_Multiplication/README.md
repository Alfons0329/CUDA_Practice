# CUDA Matrix multiplication (support both rectangular and square matrix)

## Usage:

```
nvcc main_1D.cu 
./a.out $row_A $col_A $col_B $thread_cnt_in_block
```

## Hardware specs
* CPU: Core i9 9900K 8C16T@3.6GHz
* GPU: NVIDIA TITAN V
* OS: Ubuntu 18.04.3 LTS
* RAM: DDR4 2400 48GB
* SSD: Samsung 970EVO Plus 500GB

## Benchmark log

```
CPU time (ms): 5231
GPU time (ms): 3
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
1024, 1024, 1024, 1024, 1743.666626

CPU time (ms): 66555
GPU time (ms): 30
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
2048, 2048, 2048, 1024, 2218.500000

CPU time (ms): 1204433
GPU time (ms): 211
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
4096, 4096, 4096, 1024, 5708.213379

CPU time (ms): 11786859
GPU time (ms): 1660
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
8192, 8192, 8192, 1024, 7100.517578

CPU time (ms): 5166
GPU time (ms): 3
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
1024, 1024, 1024, 1024, 1722.000000

CPU time (ms): 10391
GPU time (ms): 7
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
1024, 1024, 2048, 1024, 1484.428589

CPU time (ms): 27142
GPU time (ms): 14
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
1024, 1024, 4096, 1024, 1938.714233

CPU time (ms): 5162
GPU time (ms): 16
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
1024, 1024, 1024, 16, 322.625000

CPU time (ms): 5177
GPU time (ms): 4
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
1024, 1024, 1024, 64, 1294.250000

CPU time (ms): 5154
GPU time (ms): 4
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
1024, 1024, 1024, 256, 1288.500000

CPU time (ms): 5164
GPU time (ms): 3
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
1024, 1024, 1024, 1024, 1721.333374


```

## Future to-do
* Shared memory architecture in square matrix
* CUDA Constant memory 
