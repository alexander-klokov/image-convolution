#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <npp.h>

#include "image_utils.h"

// Helper function to handle CUDA errors
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__          \
                      << " - " << cudaGetErrorString(err) << std::endl;           \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Helper function to handle NPP errors
#define NPP_CHECK(call)                                                           \
    do {                                                                          \
        NppStatus status = call;                                                  \
        if (status != NPP_SUCCESS) {                                              \
            std::cerr << "NPP Error at " << __FILE__ << ":" << __LINE__           \
                      << " - " << nppGetErrorString(status) << std::endl;         \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

int main(int argc, char** argv) {

    std::string inputFilename = argv[1];
    std::string outputFilename = argv[2];

    Image inputImage = loadImage(inputFilename);
    if (inputImage.channels != 1) {
        std::cerr << "Error: This naive convolution code expects a grayscale (P5) image. Please provide a .pgm file." << std::endl;
        return 1;
    }
    std::cout << "Loaded image: " << inputFilename << " (" << inputImage.width << "x" << inputImage.height << ", " << inputImage.channels << " channels)\n";

    // Prepare output image buffer
    Image outputImage = inputImage; // Copy metadata, data will be overwritten
    outputImage.data.assign(inputImage.data.size(), 0); // Initialize with zeros

    // 1. Define image dimensions and convolution filter
    const int width = inputImage.width;
    const int height = inputImage.height;
    const int image_size_bytes = width * height * sizeof(Npp8u);

    // Define the convolution kernel (3x3 box filter)
    // The filter values are normalized later by the NPP function
    const Npp32s filter_size = 41;
    const Npp32s filter_dim = filter_size * filter_size;
    std::vector<Npp32s> h_kernel(filter_dim, 1 / (41.f * 41.f));

    // NPP requires a kernel size, which is half the filter dimension on each side
    NppiSize oKernelSize = {filter_size, filter_size};

    // 2. Allocate device memory
    Npp8u* d_src = nullptr;
    Npp8u* d_dst = nullptr;
    Npp32s* d_kernel = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_src, image_size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_dst, image_size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_kernel, filter_dim * sizeof(Npp32s)));

    // 3. Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_src, inputImage.data.data(), image_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(), filter_dim * sizeof(Npp32s), cudaMemcpyHostToDevice));

    // 4. Set up image and kernel data structures
    NppiSize oSrcSize = {width, height};
    NppiPoint oSrcOffset = {0, 0};
    Npp32s nDivisor = filter_dim; // Normalization divisor

    // 5. Call the NPP convolution function
    // NppiFilter_8u_C1R: 8-bit unsigned, 1 channel, ROI (Region of Interest)
  
    nppiFilter_8u_C1R(d_src, 1,
                      d_dst, 1, 
                      oSrcSize, 
                      d_kernel, 
                      oKernelSize, 
                      oSrcOffset, 
                      nDivisor);

    // 6. Copy the result back from device to host
    CUDA_CHECK(cudaMemcpy(outputImage.data.data(), d_src, image_size_bytes, cudaMemcpyDeviceToHost));
    saveImage(outputFilename, outputImage);

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_kernel));

    std::cout << "NPP convolution successful!" << std::endl;

    return 0;
}