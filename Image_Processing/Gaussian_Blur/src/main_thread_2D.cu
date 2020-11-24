/*This version of CUDA code does not use cudaMallocHost for async IO acceleration */
#include "../utils/img_io.cpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <bits/stdc++.h>
#include <sys/time.h>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

using namespace std;

// CUDA Stream
#define N_STREAMS 10

// Gaussian filter
int filter_size;
unsigned int filter_scale, filter_row;
unsigned int *filter;

// Image IO
unsigned char *img_input;
unsigned char *img_output;

// Global data
int thread_cnt, block_row;
int cudaError_cnt;
string img_name;

// CUDA error checker
void cuda_err_chk(const cudaError_t& e, const int& cudaError_cnt){
    if(e != cudaSuccess){
        fprintf(stderr, "cudaError in no. %d: %s\n", cudaError_cnt, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

// Kernel, 1D dim and grid configuration version
__global__ void cuda_gaussian_filter_thread_2D(unsigned char* img_input_cuda, unsigned char* img_output_cuda, int img_row, int img_col, int shift, unsigned int* filter_cuda, int filter_row, unsigned int filter_scale, int img_border){
    int cuda_col = blockIdx.x * blockDim.x + threadIdx.x;
    int cuda_row = blockIdx.y * blockDim.y + threadIdx.y;
    
    unsigned int tmp = 0;
    int target = 0;
    int a, b;
    
    if (3 * (cuda_row * img_col + cuda_col) + shift >= img_border){
        return;
    }
    
    for(int j = 0; j < filter_row; j++){
        for(int i = 0; i < filter_row; i++){
            a = cuda_col + i - (filter_row / 2);
            b = cuda_row + j - (filter_row / 2);
            
            target = 3 * (b * img_col + a) + shift;
            if (target >= img_border || target < 0){
                continue;
            }
			tmp += filter_cuda[j * filter_row + i] * img_input_cuda[target];  
        }
    }
    tmp /= filter_scale;
    
    if(tmp > 255){
        tmp = 255;
    }
    
    img_output_cuda[3 * (cuda_row * img_col + cuda_col) + shift] = tmp;
}

int cuda_run(const int& img_row, const int& img_col, const int& resolution, const int& async){
    /*-------------- CUDA init ------------*/
    // Allocate memory
    img_output = new unsigned char[resolution];
    memset(img_output, 0, sizeof(unsigned char) * resolution);

    unsigned char* img_input_cuda;
    unsigned char* img_output_cuda;
    unsigned int* filter_cuda;
    cuda_err_chk(cudaMalloc((void**) &img_input_cuda, resolution * sizeof(unsigned char)), cudaError_cnt++);
    cuda_err_chk(cudaMalloc((void**) &img_output_cuda, resolution * sizeof(unsigned char)), cudaError_cnt++);
    cuda_err_chk(cudaMalloc((void**) &filter_cuda, filter_size * sizeof(unsigned int)), cudaError_cnt++);
    
    // Copy memory from host to GPU
    cuda_err_chk(cudaMemcpy(filter_cuda, filter, filter_size * sizeof(unsigned int), cudaMemcpyHostToDevice), cudaError_cnt++); 
    
    // Thread configurations
    const dim3 block_size(block_row, block_row);
    const dim3 grid_size_sync((img_col + block_row - 1) / block_row, (img_row + block_row - 1) / (block_row));
    const dim3 grid_size_async((img_col + block_row - 1) / block_row, (img_row / N_STREAMS + block_row - 1) / (block_row));
    
    // Init CUDA streams
    int offset = 0;
    int chunk_size = resolution / N_STREAMS;
    cudaStream_t streams[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Measuring time
    struct timeval start, end; 
    gettimeofday(&start, 0);
    if(async){
        /*-------------- CUDA run async ------------*/
        // for(int j = 0; j < N_STREAMS; j++){
        //     offset = chunk_size * j;
            
        //     cuda_err_chk(cudaMemcpyAsync(img_input_cuda + offset, img_input + offset, chunk_size * sizeof(unsigned char), cudaMemcpyHostToDevice, streams[j]), cudaError_cnt++);
            
        //     for(int i = 0; i < 3; i++) {
        //         cuda_gaussian_filter_thread_2D<<<grid_size_async, block_size, 0, streams[j]>>>(img_input_cuda + offset, img_output_cuda + offset, img_row, img_col, i, filter_cuda, filter_row, filter_scale, chunk_size); 
        //     }
            
        //     cuda_err_chk(cudaMemcpyAsync(img_output + offset, img_output_cuda + offset, chunk_size * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[j]), cudaError_cnt++);
        // }

        for(int j = 0; j < N_STREAMS; j++){
            offset = chunk_size * j;
            
            cuda_err_chk(cudaMemcpyAsync(img_input_cuda + offset, img_input + offset, chunk_size * sizeof(unsigned char), cudaMemcpyHostToDevice, streams[j]), cudaError_cnt++);
        }

        for(int j = 0; j < N_STREAMS; j++){
            offset = chunk_size * j;
            
            for(int i = 0; i < 3; i++) {
                cuda_gaussian_filter_thread_2D<<<grid_size_async, block_size, 0, streams[j]>>>(img_input_cuda + offset, img_output_cuda + offset, img_row, img_col, i, filter_cuda, filter_row, filter_scale, chunk_size); 
            }
        }

        for(int j = 0; j < N_STREAMS; j++){
            offset = chunk_size * j;
            
            cuda_err_chk(cudaMemcpyAsync(img_output + offset, img_output_cuda + offset, chunk_size * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[j]), cudaError_cnt++);
        }
        
    }
    else{
        /*-------------- CUDA run sync ------------*/
        cuda_err_chk(cudaMemcpy(img_input_cuda, img_input, resolution * sizeof(unsigned char), cudaMemcpyHostToDevice), cudaError_cnt++);
        for(int i = 0; i < 3; i++) {
            cuda_gaussian_filter_thread_2D<<<grid_size_sync, block_size>>>(img_input_cuda, img_output_cuda, img_row, img_col, i, filter_cuda, filter_row, filter_scale, resolution);
        }
        cuda_err_chk(cudaMemcpy(img_output, img_output_cuda, resolution * sizeof(unsigned char), cudaMemcpyDeviceToHost), cudaError_cnt++);
    }
    cuda_err_chk(cudaDeviceSynchronize(), cudaError_cnt++);

    gettimeofday(&end, 0);
    int sec = end.tv_sec - start.tv_sec;
    int usec = end.tv_usec - start.tv_usec;
    int t_gpu = sec * 1000 + (usec / 1000);
    printf(ANSI_COLOR_RED "GPU time (ms): %d " ANSI_COLOR_RESET "\n", t_gpu);
    img_write(img_name, img_output, img_row, img_col, 2, async);
    
    cuda_err_chk(cudaFree(img_input_cuda), cudaError_cnt++);
    cuda_err_chk(cudaFree(img_output_cuda), cudaError_cnt++);
    cuda_err_chk(cudaFree(filter_cuda), cudaError_cnt++);
    printf("[LOG]: Finished %s\n\n", async ? "async" : "sync");
    
    return t_gpu;
}

void free_memory(){
    free(img_input);
    free(img_output);
    free(filter);
}

int main(int argc, char* argv[]){
    /*--------------- Init -------------------*/
    cudaError_cnt = 0;
    if(argc < 2){
        fprintf(stderr, "%s", "Please provide filename for Gaussian Blur. usage ./gb_thread_1D.o <Image file> \n");
        return -1;
    }
    else if(argc == 3){
        sscanf(argv[2], "%d", &thread_cnt);        
    }
    else{
        // Set default thread count to 1024
        thread_cnt = 1024;
    }
    block_row = (int)sqrt(thread_cnt);
    
    int num = 0;
    cudaGetDeviceCount(&num);
    cudaDeviceProp prop;
    if(num > 0){
        cudaGetDeviceProperties(&prop, 0);
        printf("[LOG]: Device: %s \n", prop.name);
    }
    else{
        fprintf(stderr, "%s", "No NVIDIA GPU detected!\n");
        return 1;
    }

    /*---------------- Image and mask IO ----*/
    int img_row, img_col, resolution;
    img_name = argv[1];
    img_input = img_read(img_name, img_row, img_col);
    resolution = 3 * img_row * img_col;
    
    FILE* mask;
    mask = fopen("mask_Gaussian.txt", "r");
    fscanf(mask, "%d", &filter_size);
    filter_row = (int)sqrt(filter_size);
    filter = new unsigned int [filter_size];
    
    for(int i = 0; i < filter_size; i++){
        fscanf(mask, "%u", &filter[i]);
    }
    
    filter_scale = 0;
    for(int i = 0; i < filter_size; i++){
        filter_scale += filter[i];	
    }
    fclose(mask);
    
    /*-------------- CUDA run ------------*/
    int t1, t2;
    t2 = cuda_run(img_row, img_col, resolution, 1);
    t1 = cuda_run(img_row, img_col, resolution, 0);
    
    printf(ANSI_COLOR_YELLOW "[RESULT]: [img_row, img_col, threads in each block]" ANSI_COLOR_RESET "\n");
    printf(ANSI_COLOR_YELLOW "[RESULT]: [stream workers, speedup ratio]" ANSI_COLOR_RESET "\n");
    printf("%d, %d, %d\n", img_row, img_col, thread_cnt);
    printf("%d, %.2f\n", N_STREAMS, ((float) t1 / (float)t2));
    
    /*-------------- Cleanup ------------*/
    free_memory();
    return 0;
}
