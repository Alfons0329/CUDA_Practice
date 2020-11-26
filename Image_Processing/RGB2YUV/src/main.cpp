#include <cuda_runtime_api.h>
#include "nvjpegDecoder.h"
#include <bits/stdc++.h>

using namespace std;

void img_read_nv(string img_name, int &img_row, int &img_col){
}

void img_write_nv(string img_name, unsigned char *img, const int &img_row, const int &img_col, const int &thread_dim, const int &async){
}

int main(){
    decode_params_t params;
    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    nvjpegPinnedAllocator_t pinned_allocator = {&host_malloc, &host_free};
    nvjpegStatus_t status = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, &dev_allocator, &pinned_allocator, NVJPEG_FLAGS_DEFAULT, &params.nvjpeg_handle);

    cudaEvent_t ev;
    cudaEventCreate(&ev);
    cudaEventCreateWithFlags(&ev, cudaEventBlockingSync);
    cout << "Name: " << typeid(decltype(params)).name()  << "\n";
    cout << "Name: " << typeid(decltype(ev)).name() << "\n";
    cout << "Finished testing " << "\n";
    return 0;
}
