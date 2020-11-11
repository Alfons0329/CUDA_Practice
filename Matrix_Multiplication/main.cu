#include <bits/stdc++.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

int row_A, col_A, col_B; 
int** mat_A;
int** mat_B;
int** mat_C;
int** mat_A_CUDA;
int** mat_B_CUDA;
int** mat_C_CUDA;

void mul_gpu(){
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

__global__ void mul_cuda(int row_A){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int sum = 0;

    if(col <)
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
            // mat_A[i][j] = mat_B[i][j]= unif(generator);
            mat[i][j] = is_C ? 0 : i * col + j;
        }
    }

    return mat;
}

int main(int argc, char* argv[]){
    if(argc != 4){
        cerr << "Usage: ./a.out $row_A $col_A $col_B " << '\n';
        exit(-1);
    }
    row_A = atoi(argv[1]);
    col_A = atoi(argv[2]);
    col_B = atoi(argv[3]);
    assert(row_A > 0 && col_A > 0 && col_B > 0);

    mat_A = init(row_A, col_A, false);
    mat_B = init(col_A, col_B, false);
    mat_C = init(row_A, col_B, true);
    mat_C_CUDA = init(row_A, col_B, true);
    
    struct timeval start, end;
    gettimeofday(&start, 0);
    mul_gpu();
    gettimeofday(&end, 0);
    int sec = end.tv_sec - start.tv_sec;
    int usec = end.tv_usec - start.tv_usec;
    printf("Serial time: %d ms\n", sec * 1000 + (usec / 1000));

    mat_C = init(row_A, col_B, true);

    cudaError_t ce_A, ce_B, ce_C;
    
    ce_A = cudaMalloc((void**) &mat_A_CUDA, row_A * col_A * sizeof(int));
    ce_B = cudaMalloc((void**) &mat_B_CUDA, col_A * col_B * sizeof(int));
    ce_C = cudaMalloc((void**) &mat_C_CUDA, row_A * col_B * sizeof(int));
    if(ce_A != cudaSuccess ||
    ce_B != cudaSuccess || 
    ce_C != cudaSuccess){
        cerr << "cudaMalloc failed \n" << '\n';
        exit(1);
    }
    
    ce_A = cudaMemcpy(mat_A_CUDA, mat_A, row_A * col_A * sizeof(int), cudaMemcpyHostToDevice);
    ce_B = cudaMemcpy(mat_B_CUDA, mat_B, col_A * col_B * sizeof(int), cudaMemcpyHostToDevice);
    ce_C = cudaMemcpy(mat_C_CUDA, mat_C, row_A * col_B * sizeof(int), cudaMemcpyHostToDevice);
    if(ce_A != cudaSuccess ||
    ce_B != cudaSuccess || 
    ce_C != cudaSuccess){
        cerr << "cudaMalloc failed \n" << '\n';
        exit(1);
    }

    printf("Check integrity \n");
    for(int i = 0; i < row_A; i++){
        for(int j = 0; j < col_B; j++){
            assert(mat_C[i][j] == mat_C_CUDA[i][j]);
        }
    }

    return 0;
}
