/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include "nvjpegDecoder.h"


int decode_images(const FileData &img_data, const std::vector<size_t> &img_len,
        std::vector<nvjpegImage_t> &out, decode_params_t &params,
        double &time) {
    CHECK_CUDA(cudaStreamSynchronize(params.stream));
    cudaEvent_t startEvent = NULL, stopEvent = NULL;
    float loopTime = 0; 

    CHECK_CUDA(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
    CHECK_CUDA(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));


    std::vector<const unsigned char*> batched_bitstreams;
    std::vector<size_t> batched_bitstreams_size;
    std::vector<nvjpegImage_t>  batched_output;

    // bit-streams that batched decode cannot handle
    std::vector<const unsigned char*> otherdecode_bitstreams;
    std::vector<size_t> otherdecode_bitstreams_size;
    std::vector<nvjpegImage_t> otherdecode_output;

    if(params.hw_decode_available){
        for(int i = 0; i < params.batch_size; i++){
            // extract bitstream meta data to figure out whether a bit-stream can be decoded
            nvjpegJpegStreamParseHeader(params.nvjpeg_handle, (const unsigned char *)img_data[i].data(), img_len[i], params.jpeg_streams[0]);
            int isSupported = -1;
            nvjpegDecodeBatchedSupported(params.nvjpeg_handle, params.jpeg_streams[0], &isSupported);

            if(isSupported == 0){
                batched_bitstreams.push_back((const unsigned char *)img_data[i].data());
                batched_bitstreams_size.push_back(img_len[i]);
                batched_output.push_back(out[i]);
            } else {
                otherdecode_bitstreams.push_back((const unsigned char *)img_data[i].data());
                otherdecode_bitstreams_size.push_back(img_len[i]);
                otherdecode_output.push_back(out[i]);
            }
        }
    } else {
        for(int i = 0; i < params.batch_size; i++) {
            otherdecode_bitstreams.push_back((const unsigned char *)img_data[i].data());
            otherdecode_bitstreams_size.push_back(img_len[i]);
            otherdecode_output.push_back(out[i]);
        }
    }

    CHECK_CUDA(cudaEventRecord(startEvent, params.stream));

    if(batched_bitstreams.size() > 0)
    {
        CHECK_NVJPEG(
                nvjpegDecodeBatchedInitialize(params.nvjpeg_handle, params.nvjpeg_state,
                    batched_bitstreams.size(), 1, params.fmt));

        CHECK_NVJPEG(nvjpegDecodeBatched(
                    params.nvjpeg_handle, params.nvjpeg_state, batched_bitstreams.data(),
                    batched_bitstreams_size.data(), batched_output.data(), params.stream));
    }

    if(otherdecode_bitstreams.size() > 0)
    {
        CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(params.nvjpeg_decoupled_state, params.device_buffer));
        int buffer_index = 0;
        CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(params.nvjpeg_decode_params, params.fmt));
        for (int i = 0; i < params.batch_size; i++) {
            CHECK_NVJPEG(
                    nvjpegJpegStreamParse(params.nvjpeg_handle, otherdecode_bitstreams[i], otherdecode_bitstreams_size[i],
                        0, 0, params.jpeg_streams[buffer_index]));

            CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(params.nvjpeg_decoupled_state,
                        params.pinned_buffers[buffer_index]));

            CHECK_NVJPEG(nvjpegDecodeJpegHost(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
                        params.nvjpeg_decode_params, params.jpeg_streams[buffer_index]));

            CHECK_CUDA(cudaStreamSynchronize(params.stream));

            CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
                        params.jpeg_streams[buffer_index], params.stream));

            buffer_index = 1 - buffer_index; // switch pinned buffer in pipeline mode to avoid an extra sync

            CHECK_NVJPEG(nvjpegDecodeJpegDevice(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
                        &otherdecode_output[i], params.stream));

        }
    }
    CHECK_CUDA(cudaEventRecord(stopEvent, params.stream));

    CHECK_CUDA(cudaEventSynchronize(stopEvent));
    CHECK_CUDA(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
    time = static_cast<double>(loopTime);

    return EXIT_SUCCESS;
}

int write_images(std::vector<nvjpegImage_t> &iout, std::vector<int> &widths,
        std::vector<int> &heights, decode_params_t &params,
        FileNames &filenames) {
    for (int i = 0; i < params.batch_size; i++) {
        // Get the file name, without extension.
        // This will be used to rename the output file.
        size_t position = filenames[i].rfind("/");
        std::string sFileName =
            (std::string::npos == position)
            ? filenames[i]
            : filenames[i].substr(position + 1, filenames[i].size());
        position = sFileName.rfind(".");
        sFileName = (std::string::npos == position) ? sFileName
            : sFileName.substr(0, position);
        std::string fname(params.output_dir + "/" + sFileName + ".bmp");

        int err;
        if (params.fmt == NVJPEG_OUTPUT_RGB || params.fmt == NVJPEG_OUTPUT_BGR) {
            err = writeBMP(fname.c_str(), iout[i].channel[0], iout[i].pitch[0],
                    iout[i].channel[1], iout[i].pitch[1], iout[i].channel[2],
                    iout[i].pitch[2], widths[i], heights[i]);
        } else if (params.fmt == NVJPEG_OUTPUT_RGBI ||
                params.fmt == NVJPEG_OUTPUT_BGRI) {
            // Write BMP from interleaved data
            err = writeBMPi(fname.c_str(), iout[i].channel[0], iout[i].pitch[0],
                    widths[i], heights[i]);
        }
        if (err) {
            std::cout << "Cannot write output file: " << fname << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "Done writing decoded image to file: " << fname << std::endl;
    }
    return EXIT_SUCCESS;
}

double process_images(FileNames &image_names, decode_params_t &params,
        double &total) {
    // vector for storing raw files and file lengths
    FileData file_data(params.batch_size);
    std::vector<size_t> file_len(params.batch_size);
    FileNames current_names(params.batch_size);
    std::vector<int> widths(params.batch_size);
    std::vector<int> heights(params.batch_size);
    // we wrap over image files to process total_images of files
    FileNames::iterator file_iter = image_names.begin();

    // stream for decoding
    CHECK_CUDA(
            cudaStreamCreateWithFlags(&params.stream, cudaStreamNonBlocking));

    int total_processed = 0;

    // output buffers
    std::vector<nvjpegImage_t> iout(params.batch_size);
    // output buffer sizes, for convenience
    std::vector<nvjpegImage_t> isz(params.batch_size);

    for (int i = 0; i < iout.size(); i++) {
        for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
            iout[i].channel[c] = NULL;
            iout[i].pitch[c] = 0;
            isz[i].pitch[c] = 0;
        }
    }

    double test_time = 0;
    int warmup = 0;
    while (total_processed < params.total_images) {
        if (read_next_batch(image_names, params.batch_size, file_iter, file_data,
                    file_len, current_names))
            return EXIT_FAILURE;

        if (prepare_buffers(file_data, file_len, widths, heights, iout, isz,
                    current_names, params))
            return EXIT_FAILURE;

        double time;
        if (decode_images(file_data, file_len, iout, params, time))
            return EXIT_FAILURE;
        if (warmup < params.warmup) {
            warmup++;
        } else {
            total_processed += params.batch_size;
            test_time += time;
        }

        if (params.write_decoded)
            write_images(iout, widths, heights, params, current_names);
    }
    total = test_time;

    release_buffers(iout);

    CHECK_CUDA(cudaStreamDestroy(params.stream));

    return EXIT_SUCCESS;
}



