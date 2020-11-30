#include "nvjpegDecoder.h"
// #include <cuda.h>
// #include <cuda_runtime.h>
#define BLOCK_SIZE 32

using namespace std;

__global__ void simple_kernel_function_RGB(unsigned char* img_R, unsigned char* img_G, unsigned char* img_B, const int img_row, const int img_col){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < img_row && col < img_col) {
        img_R[row * img_col + col] = 0;
        img_G[row * img_col + col] = 255;
        img_B[row * img_col + col] = 0;
    }
}

void image_processing_gpu(vector<nvjpegImage_t> &iout, vector<int> &widths, vector<int> &heights, decode_params_t &params){
    unsigned char* img_RGB = NULL; // pointer to the RGB 3 channel obj, RGBI (interleaving) mode
    unsigned char* img_R = NULL; // pointer to the R 3 channel obj
    unsigned char* img_G = NULL; // pointer to the R 3 channel obj
    unsigned char* img_B = NULL; // pointer to the R 3 channel obj

    for(int batch = 0; batch < params.batch_size; batch++){
        int img_row = heights[batch];
        int img_col = widths[batch];
        int grid_row = (img_row + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int grid_col = (img_col + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid(grid_col, grid_row);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        if(params.fmt == NVJPEG_OUTPUT_RGB){
            img_R = iout[batch].channel[0];
            img_G = iout[batch].channel[1];
            img_B = iout[batch].channel[2];
            if(img_R == NULL || img_G == NULL || img_B == NULL){
                fprintf(stderr, "%s", "Nullpointerexception \n");
            }
            simple_kernel_function_RGB<<<dimGrid, dimBlock>>>(img_R, img_G, img_B, img_row, img_col);
            printf("Finished GPU kernel call \n");
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
            printf("no-operations\n");
            // nop
        }
    }
    printf("Finished image processing in GPU \n");
}

int image_processing_cpu(vector<nvjpegImage_t> &iout, vector<int> &widths, vector<int> &heights, decode_params_t &params){
    unsigned char* img_RGB = NULL; // pointer to the RGB 3 channel obj, RGBI (interleaving) mode
    unsigned char* img_R = NULL; // pointer to the R 3 channel obj
    unsigned char* img_G = NULL; // pointer to the R 3 channel obj
    unsigned char* img_B = NULL; // pointer to the R 3 channel obj

    cout << "Start image procesing " << endl;
    for(int batch = 0; batch < params.batch_size; batch++){
        int img_row = heights[batch];
        int img_col = widths[batch];

        cout << "Process image #" << batch << " img_row " << img_row << " img_col " << img_col << endl;
        if(params.fmt == NVJPEG_OUTPUT_RGB){
            img_R = iout[batch].channel[0];
            img_G = iout[batch].channel[1];
            img_B = iout[batch].channel[2];
            cout << "RGB " << endl;
            if(img_R == NULL || img_G == NULL || img_B == NULL){
                cerr << "NULLPointerException " << endl;
                return EXIT_FAILURE;
            }

            for(int i = 0; i < img_row / 2; i++){
                for(int j = 0; j < img_col; j++){
                    swap(img_R[i * img_col + j], img_R[(img_row - i) * img_col + j]);
                    swap(img_G[i * img_col + j], img_G[(img_row - i) * img_col + j]);
                    swap(img_B[i * img_col + j], img_B[(img_row - i) * img_col + j]);
                }
            }
        }
        else if(params.fmt == NVJPEG_OUTPUT_BGR){
            img_R = iout[batch].channel[2];
            img_G = iout[batch].channel[1];
    img_B = iout[batch].channel[0];
            cout << "BGR " << endl;
            for(int i = 0; i < img_row / 2; i++){
                for(int j = 0; j < img_col; j++){
                    cout << "i " << i << ", j " << j << endl;
                    swap(img_R[i * img_col + j], img_R[(img_row - i) * img_col + j]);
                    swap(img_G[i * img_col + j], img_G[(img_row - i) * img_col + j]);
                    swap(img_B[i * img_col + j], img_B[(img_row - i) * img_col + j]);
                }
            }
        }
        else if(params.fmt == NVJPEG_OUTPUT_RGBI || params.fmt == NVJPEG_OUTPUT_BGRI){
            img_RGB = iout[batch].channel[0];
            cout << "RGBi / BGRi " << endl;
            for(int i = 0; i < img_row / 2; i++){
                for(int j = 0; j < img_col; j++){
                    cout << "i " << i << ", j " << j << endl;
                    swap(img_RGB[3 * (i * img_col + j) + 0], img_RGB[3 * ((img_row - i) * img_col + j) + 0]);
                    swap(img_RGB[3 * (i * img_col + j) + 1], img_RGB[3 * ((img_row - i) * img_col + j) + 1]);
                    swap(img_RGB[3 * (i * img_col + j) + 2], img_RGB[3 * ((img_row - i) * img_col + j) + 2]);
                }
            }
        }
        else{
            cerr << "Unsuppported image codec! " << endl;
            return EXIT_FAILURE;
        }
    }

    cout << "Finished image processing successfully! " << endl;
    return EXIT_SUCCESS;
}
