#include <bits/stdc++.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

void mul_cpu(int row_A, int col_A, int col_B, int** mat_A, int** mat_B, int** mat_C){
    for(int i = 0; i < row_A; i++){
        for(int j = 0; j < col_B; j++){
            for(int k = 0; k < col_A; k++){
                // printf("%d x %d, ", mat_A[i][k], mat_B[k][j]);
                mat_C[i][j] += mat_A[i][k] * mat_B[k][j];
            }
            // printf("\n");
        }
    }
}

__global__ void mul_cuda(int row_A, int col_A, int col_B, int** mat_A_CUDA, int** mat_B_CUDA, int** mat_C_CUDA){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < row_A && col < col_B && row >= 0 && col >= 0){
        for(int k = 0; k < col_A; k++){
            mat_C_CUDA[row][col] += mat_A_CUDA[row][k] * mat_B_CUDA[k][col];
        }        
        printf("GPU row %d col %d res %d\n", row, col, mat_C_CUDA[row][col]);
    }
}


int** init(int row, int col, bool is_C){
    int** mat = (int**) malloc(row * sizeof(int *));
    for(int i = 0; i < row; i++){
        mat[i] = (int *) malloc(col * sizeof(int));
    }

    random_device rd;
    mt19937 generator(rd());
    uniform_int_distribution<int> unif(INT_MIN, INT_MAX);

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            mat[i][j] = is_C ? 0 : unif(generator);
        }
    }

    return mat;
}

int main(int argc, char* argv[]){
    /*-------------- CPU init ------------*/
    int row_A, col_A, col_B;
    int** mat_A;
    int** mat_B;
    int** mat_C;
    int** mat_A_CUDA;
    int** mat_B_CUDA;
    int** mat_C_CUDA;
    int** res_CPU;
    int** res_GPU;

    if(argc != 5){
        cerr << "Usage: ./a.out $row_A $col_A $col_B BLOCK_SIZE in 1Dim direction " << '\n';
        exit(-1);
    }

    row_A = atoi(argv[1]);
    col_A = atoi(argv[2]);
    col_B = atoi(argv[3]);
    assert(row_A > 0 && col_A > 0 && col_B > 0);

    mat_A = init(row_A, col_A, false);
    mat_B = init(col_A, col_B, false);
    mat_C = init(row_A, col_B, true);
    res_CPU = init(row_A, col_B, true);

    /*-------------- CPU run -------------*/
    struct timeval start, end;
    gettimeofday(&start, 0);
    mul_cpu(row_A, col_A, col_B, mat_A, mat_B, mat_C);
    gettimeofday(&end, 0);
    int sec = end.tv_sec - start.tv_sec;
    int usec = end.tv_usec - start.tv_usec;
    cout << "CPU serial time (ms): " << sec * 1000 + (usec / 1000) << '\n';

    res_CPU = init(row_A, col_B, true);
    for(int i = 0; i < row_A; i++){
        for(int j = 0; j < col_B; j++){
            res_CPU[i][j] = mat_C[i][j];
            mat_C[i][j] = 0;
        }
    }
    /*-------------- CUDA init ------------*/
    cudaError_t ce_A, ce_B, ce_C;
    printf("%d %d %d \n", row_A, col_A, col_B);

    ce_A = cudaMalloc((void**) &mat_A_CUDA, row_A * col_A * sizeof(int));
    ce_B = cudaMalloc((void**) &mat_B_CUDA, col_A * col_B * sizeof(int));
    ce_C = cudaMalloc((void**) &mat_C_CUDA, row_A * col_B * sizeof(int));
    if( ce_A != cudaSuccess ||
        ce_B != cudaSuccess || 
        ce_C != cudaSuccess){
        cerr << "cudaMalloc failed " << '\n';
        exit(1);
    }

    ce_A = cudaMemcpy(mat_A_CUDA, mat_A, row_A * col_A * sizeof(int), cudaMemcpyHostToDevice);
    ce_B = cudaMemcpy(mat_B_CUDA, mat_B, col_A * col_B * sizeof(int), cudaMemcpyHostToDevice);
    ce_C = cudaMemcpy(mat_C_CUDA, mat_C, row_A * col_B * sizeof(int), cudaMemcpyHostToDevice);
    if( ce_A != cudaSuccess ||
        ce_B != cudaSuccess || 
        ce_C != cudaSuccess){
        cerr << "cudaMemcpyHostToDevice failed " << '\n';
        exit(2);
    }

    const int BLOCK_SIZE = atoi(argv[4]);
    // configured as method 2 in https://blog.csdn.net/yongjiankuang/article/details/90180559
    const dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 dim_grid((row_A + BLOCK_SIZE + 1) / BLOCK_SIZE,(col_B + BLOCK_SIZE + 1) / BLOCK_SIZE);

    /*-------------- CUDA run -------------*/
    gettimeofday(&start, 0);
    mul_cuda<<<dim_grid, dim_block, 0>>>(row_A, col_A, col_B, mat_A_CUDA, mat_B_CUDA, mat_C_CUDA);
    cudaError_t ce_K; // cuda erroe for kernel
    ce_K = cudaDeviceSynchronize();
    if(ce_K != cudaSuccess){
        cerr << "cudaDeviceSynchronize failed " << '\n';
    }
    gettimeofday(&end, 0);
    sec = end.tv_sec - start.tv_sec;
    usec = end.tv_usec - start.tv_usec;
    cout << "GPU CUDA time (ms): " << sec * 1000 + (usec / 1000) << '\n';

    /*------- Check integrity -------------*/
    res_GPU = init(row_A, col_B, true);
    ce_C = cudaMemcpy(res_GPU, mat_C_CUDA, row_A * col_B * sizeof(int), cudaMemcpyDeviceToHost);
    if(ce_C != cudaSuccess){
        cerr << "cudaMemcpyDeviceToHost failed " << '\n';
        exit(4);
    }

    cout << "Check integrity " << '\n';
    for(int i = 0; i < row_A; i++){
        for(int j = 0; j < col_B; j++){
            printf("%d, %d |", res_CPU[i][j], res_GPU[i][j]);
            assert(res_CPU[i][j] == res_GPU[i][j]);
        }
        printf("\n");
    }
    cout << "Finished all" << '\n';

    /*------- Clear memory -------------*/
    cudaFree(mat_A_CUDA);
    cudaFree(mat_B_CUDA);
    cudaFree(mat_C_CUDA);
    free(mat_A);
    free(mat_B);
    free(mat_C);

    return 0;
}
