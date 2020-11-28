using namespace std;

int image_prcoessing(vector<nvjpegImage_t> &iout, vector<int> &widths, vector<int> &heights, decode_params_t &param){
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
            for(int i = 0; i < img_row / 2; i++){
                for(int j = 0; j < img_col; j++){
                    swap(img_R[i * img_col + j], img_R[(img_row - i) * img_col + j]);
                    swap(img_G[i * img_col + j], img_G[(img_row - i) * img_col + j]);
                    swap(img_B[i * img_col + j], img_B[(img_row - i) * img_col + j]);
                }
            }
        }
        else if(params.fmt == NVJPEG_OUTPUT_RGBI || params.fmt == NVJPEG_OUTPUT_BGRI){
            img_RGB = iout[batch];
            for(int i = 0; i < img_row / 2; i++){
                for(int j = 0; j < img_col; j++){
                    swap(img_RGB[3 * (i * img_col + j) + 0], img_RGB[3 * ((img_row - i) * img_col + j) + 0]);
                    swap(img_RGB[3 * (i * img_col + j) + 1], img_RGB[3 * ((img_row - i) * img_col + j) + 1]);
                    swap(img_RGB[3 * (i * img_col + j) + 2], img_RGB[3 * ((img_row - i) * img_col + j) + 2]);
                }
            }
        }
        else{
            cerr << "Unsuppported file format! " << endl;
            return EXIT_FAILURE;
        }
    }

    cout << "Finished image processing " << endl;
    return EXIT_SUCCESS;
}
