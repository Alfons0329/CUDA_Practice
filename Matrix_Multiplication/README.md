# CUDA Matrix multiplication (support)

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
CPU time (ms): 5479
GPU time (ms): 3
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
1024, 1024, 1024, 1024, 1826.333374

CPU time (ms): 63077
GPU time (ms): 28
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
2048, 2048, 2048, 1024, 2252.750000

CPU time (ms): 1208120
GPU time (ms): 214
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
4096, 4096, 4096, 1024, 5645.420410

CPU time (ms): 5134
GPU time (ms): 3
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
1024, 1024, 1024, 1024, 1711.333374

CPU time (ms): 10396
GPU time (ms): 7
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
1024, 1024, 2048, 1024, 1485.142822

CPU time (ms): 26328
GPU time (ms): 14
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
1024, 1024, 4096, 1024, 1880.571411

CPU time (ms): 5149
GPU time (ms): 16
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
1024, 1024, 1024, 16, 321.812500

CPU time (ms): 5128
GPU time (ms): 4
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
1024, 1024, 1024, 64, 1282.000000

CPU time (ms): 5133
GPU time (ms): 4
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
1024, 1024, 1024, 256, 1283.250000

CPU time (ms): 5135
GPU time (ms): 3
Check integrity
Integrity pass!, CPU result == GPU result, all finished
[row_A, col_A, col_B, block_size(thread cnt), Accelerate ratio (times)]:
1024, 1024, 1024, 1024, 1711.666626
```

## Future to-do
* Shared memory architecture in square matrix
* CUDA Constant memory 
