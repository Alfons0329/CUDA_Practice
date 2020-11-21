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

// Gaussian filter
int filter_size;
unsigned int filter_scale, filter_row;
unsigned int *filter;

// Image IO
unsigned char *img_input;
unsigned char *img_output;

// Global data
int thread_cnt, block_row;
string img_name;

int main(int argc, char* argv[]){
    /*--------------- Init -------------------*/
    if(argc < 2){
        fprintf(stderr, "%s", "Please provide filename for Gaussian Blur. usage ./gb_thread_1D.o <Image file> \n");
        return -1;
    }
    else if(argc == 3){
        sscanf(argv[2], "%d", &thread_cnt);        
        printf("Testing with %d threads in each CUDA block\n", thread_cnt);
    }
    else{
        // Set default thread count to 1024
        thread_cnt = 1024;
    }
    block_row = (int)sqrt(thread_cnt);

    /*---------------- Image and mask IO ----*/
    int img_row, img_col;
    img_name = argv[1];
    img_input = img_read(img_name, img_row, img_col);

    img_write(img_name, img_input, img_row, img_col, 1);
    printf("Finished all \n");
    return 0;
}
