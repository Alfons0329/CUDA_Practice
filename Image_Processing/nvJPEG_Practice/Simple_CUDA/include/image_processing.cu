#include "nvjpegDecoder.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
template <typename T> __device__ void inline swap_gpu(T& a, T& b){
    T c(a);
    a = b;
    b = c;
} 
__global__ void simple_kernel_function_RGB(unsigned char* img_R, unsigned char* img_G, unsigned char* img_B, const int img_row, const int img_col){
    for(int i = 0; i < img_row / 2; i++){
        for(int j = 0; j < img_col; j++){
            swap_gpu(img_R[i * img_col + j], img_R[(img_row - i) * img_col + j]);
            swap_gpu(img_G[i * img_col + j], img_G[(img_row - i) * img_col + j]);
            swap_gpu(img_B[i * img_col + j], img_B[(img_row - i) * img_col + j]);
        }
    }
}

int image_processing_gpu(vector<nvjpegImage_t> &iout, vector<int> &widths, vector<int> &heights, decode_params_t &params){
    unsigned char* img_RGB = NULL; // pointer to the RGB 3 channel obj, RGBI (interleaving) mode
    unsigned char* img_R = NULL; // pointer to the R 3 channel obj
    unsigned char* img_G = NULL; // pointer to the R 3 channel obj
    unsigned char* img_B = NULL; // pointer to the R 3 channel obj

    for(int batch = 0; batch < params.batch_size; batch++){
        int img_row = heights[batch];
        int img_col = widths[batch];
        printf("Process image # %d\n", batch);
        if(params.fmt == NVJPEG_OUTPUT_RGB){
            printf("RGB image \n");
            img_R = iout[batch].channel[0];
            img_G = iout[batch].channel[1];
            img_B = iout[batch].channel[2];
            simple_kernel_function_RGB<<<1, 1>>>(img_R, img_G, img_B, img_row, img_col);
        }
        else if(params.fmt == NVJPEG_OUTPUT_BGR){
            img_R = iout[batch].channel[2];
            img_G = iout[batch].channel[1];
            img_B = iout[batch].channel[0];
            printf("BGR image \n");
            simple_kernel_function_RGB<<<1, 1>>>(img_B, img_G, img_R, img_row, img_col);
        }
        else if(params.fmt == NVJPEG_OUTPUT_RGBI || params.fmt == NVJPEG_OUTPUT_BGRI){
            printf("RGBi / BGRi image \n");
            img_RGB = iout[batch].channel[0];
            // simple_kernel_function_RGBI();
        }
        else{
            return EXIT_FAILURE;
        }
        printf("Finished process image # %d\n", batch);
    }
    return EXIT_SUCCESS;
}
