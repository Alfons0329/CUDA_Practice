#include <cuda_runtime_api.h>
#include "nvjpegDecoder.cpp"

using namespace std;

int main(int argc, const char *argv[]){
    int pidx;

    if ((pidx = findParamIndex(argv, argc, "-h")) != -1 ||
        (pidx = findParamIndex(argv, argc, "--help")) != -1){
        std::cout << "Usage: " << argv[0]
                  << " -i images_dir [-b batch_size] [-t total_images] "
                     "[-w warmup_iterations] [-o output_dir] "
                     "[-pipelined] [-batched] [-fmt output_format]\n";
        std::cout << "Parameters: " << std::endl;
        std::cout << "\timages_dir\t:\tPath to single image or directory of images"
                  << std::endl;
        std::cout << "\tbatch_size\t:\tDecode images from input by batches of "
                     "specified size"
                  << std::endl;
        std::cout << "\ttotal_images\t:\tDecode this much images, if there are "
                     "less images \n"
                  << "\t\t\t\t\tin the input than total images, decoder will loop "
                     "over the input"
                  << std::endl;
        std::cout << "\twarmup_iterations\t:\tRun this amount of batches first "
                     "without measuring performance"
                  << std::endl;
        std::cout
            << "\toutput_dir\t:\tWrite decoded images as BMPs to this directory"
            << std::endl;
        std::cout << "\tpipelined\t:\tUse decoding in phases" << std::endl;
        std::cout << "\tbatched\t\t:\tUse batched interface" << std::endl;
        std::cout << "\toutput_format\t:\tnvJPEG output format for decoding. One "
                     "of [rgb, rgbi, bgr, bgri, yuv, y, unchanged]"
                  << std::endl;
        return EXIT_SUCCESS;
    }

    decode_params_t params;

    params.input_dir = "./";
    if ((pidx = findParamIndex(argv, argc, "-i")) != -1){
        params.input_dir = argv[pidx + 1];
    }
    else{
        // Search in default paths for input images.
        int found = getInputDir(params.input_dir, argv[0]);
        if (!found){
            std::cout << "Please specify input directory with encoded images" << std::endl;
            return EXIT_FAILURE;
        }
    }

    params.batch_size = 1;
    if ((pidx = findParamIndex(argv, argc, "-b")) != -1){
        params.batch_size = std::atoi(argv[pidx + 1]);
    }

    params.total_images = -1;
    if ((pidx = findParamIndex(argv, argc, "-t")) != -1){
        params.total_images = std::atoi(argv[pidx + 1]);
    }

    params.warmup = 0;
    if ((pidx = findParamIndex(argv, argc, "-w")) != -1){
        params.warmup = std::atoi(argv[pidx + 1]);
    }

    params.fmt = NVJPEG_OUTPUT_RGB;
    if ((pidx = findParamIndex(argv, argc, "-fmt")) != -1){
        std::string sfmt = argv[pidx + 1];
        if (sfmt == "rgb")
            params.fmt = NVJPEG_OUTPUT_RGB;
        else if (sfmt == "bgr")
            params.fmt = NVJPEG_OUTPUT_BGR;
        else if (sfmt == "rgbi")
            params.fmt = NVJPEG_OUTPUT_RGBI;
        else if (sfmt == "bgri")
            params.fmt = NVJPEG_OUTPUT_BGRI;
        else if (sfmt == "yuv")
            params.fmt = NVJPEG_OUTPUT_YUV;
        else if (sfmt == "y")
            params.fmt = NVJPEG_OUTPUT_Y;
        else if (sfmt == "unchanged")
            params.fmt = NVJPEG_OUTPUT_UNCHANGED;
        else{
            std::cout << "Unknown format: " << sfmt << std::endl;
            return EXIT_FAILURE;
        }
    }

    params.write_decoded = false;
    if ((pidx = findParamIndex(argv, argc, "-o")) != -1)
    {
        params.output_dir = argv[pidx + 1];
        if (params.fmt != NVJPEG_OUTPUT_RGB && params.fmt != NVJPEG_OUTPUT_BGR &&
            params.fmt != NVJPEG_OUTPUT_RGBI && params.fmt != NVJPEG_OUTPUT_BGRI){
            std::cout << "We can write ony BMPs, which require output format be "
                         "either RGB/BGR or RGBi/BGRi"
                      << std::endl;
            return EXIT_FAILURE;
        }
        params.write_decoded = true;
    }

    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    nvjpegPinnedAllocator_t pinned_allocator = {&host_malloc, &host_free};

    nvjpegStatus_t status = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, &dev_allocator,
                                           &pinned_allocator, NVJPEG_FLAGS_DEFAULT, &params.nvjpeg_handle);
    params.hw_decode_available = true;
    if (status == NVJPEG_STATUS_ARCH_MISMATCH){
        std::cout << "Hardware Decoder not supported. Falling back to default backend" << std::endl;
        CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator,
                                    &pinned_allocator, NVJPEG_FLAGS_DEFAULT, &params.nvjpeg_handle));
        params.hw_decode_available = false;
    }
    else{
        CHECK_NVJPEG(status);
    }

    CHECK_NVJPEG(
        nvjpegJpegStateCreate(params.nvjpeg_handle, &params.nvjpeg_state));

    create_decoupled_api_handles(params);

    // read source images
    FileNames image_names;
    readInput(params.input_dir, image_names);

    if (params.total_images == -1){
        params.total_images = image_names.size();
    }
    else if (params.total_images % params.batch_size){
        params.total_images =
            ((params.total_images) / params.batch_size) * params.batch_size;
        std::cout << "Changing total_images number to " << params.total_images
                  << " to be multiple of batch_size - " << params.batch_size
                  << std::endl;
    }

    std::cout << "Decoding images in directory: " << params.input_dir
              << ", total " << params.total_images << ", batchsize "
              << params.batch_size << std::endl;

    double total;
    if (process_images(image_names, params, total))
        return EXIT_FAILURE;
    std::cout << "Total decoding time: " << total << std::endl;
    std::cout << "Avg decoding time per image: " << total / params.total_images
              << std::endl;
    std::cout << "Avg images per sec: " << params.total_images / total
              << std::endl;
    std::cout << "Avg decoding time per batch: "
              << total / ((params.total_images + params.batch_size - 1) /
                          params.batch_size)
              << std::endl;

    destroy_decoupled_api_handles(params);

    CHECK_NVJPEG(nvjpegJpegStateDestroy(params.nvjpeg_state));
    CHECK_NVJPEG(nvjpegDestroy(params.nvjpeg_handle));

    return EXIT_SUCCESS;
}

/*
Unused

void img_write_nv(string img_name, unsigned char *img, const int &img_row, const int &img_col, const int &thread_dim, const int &async){

}

void nvjpeg_err_chk(const nvjpegStatus_t &e){
    if (e != NVJPEG_STATUS_SUCCESS){
        fprintf(stderr, "nvjpegError code: %d\n", e);
        exit(EXIT_FAILURE);
    }
}

int img_read_nv(decode_params_t& params, vector<string> &img_names, vector<size_t> &file_lens, int &img_row, int &img_col){
    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    nvjpegPinnedAllocator_t pinned_allocator = {&host_malloc, &host_free};

    int channels;
    int img_rows[NVJPEG_MAX_COMPONENT];
    int img_cols[NVJPEG_MAX_COMPONENT];
    nvjpegChromaSubsampling_t subsampling;
    vector<nvjpegImage_t> iout(img_names.size());
    vector<nvjpegImage_t> isz(img_names.size());

    for(int i = 0; i < img_names.size(); i++){
        nvjpeg_err_chk(nvjpegGetImageInfo(params.nvjpeg_handle, (const unsigned char *)img_names[i].data(), file_lens[i], &channels, &subsampling, img_rows, img_cols));
        printf("Img with %d channels \n", channels);
        for(int c = 0; c < channels; c++){
            printf("Channel #%d is %d row x %d col \n", c, img_rows[c], img_cols[c]);
        }
        nvjpeg_err_chk(nvjpegDecode(params.nvjpeg_handle, params.nvjpeg_state, (const unsigned char *)img_names[i].data(), file_lens[i], params.fmt, &iout[i], params.stream));
        printf("Finished decode img no %d \n", i);

        for(int i = 0; i < )
    }
}

*/
