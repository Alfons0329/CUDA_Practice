#include "nvjpegDecoder.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void simple_kernel_function_RGB(unsigned char* img_R, unsigned char* img_G, unsigned char* img_B, const int& img_row, const int& img_col){
    printf("CUDA kernel function, img_row %d img_col %d \n", img_row, img_col);
    for(int i = 0; i < img_row; i++){
        for(int j = 0; j < img_col; j++){
            printf("[KERNEL]: Access i %d j %d \n", i, j);
            img_R[i * img_col + j] = 0;
            img_G[i * img_col + j] = 255;
            img_B[i * img_col + j] = 255;
        }
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

        if(params.fmt == NVJPEG_OUTPUT_RGB){
            img_R = iout[batch].channel[0];
            img_G = iout[batch].channel[1];
            img_B = iout[batch].channel[2];
            printf("RGB image! \n");
            if(!img_R || !img_G || !img_B){
                fprintf(stderr, "%s", "Nullpointerexception \n");
            }
            simple_kernel_function_RGB<<<1, 1>>>(img_R, img_G, img_B, img_row, img_col);
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