#include <iostream>
#include <chrono> 
#include <algorithm>

#include <cuda_runtime.h>

#include "image_utils.h"
#include "kernels.cuh"

// Helper macro for checking CUDA errors
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " - " << func << " failed with error " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// --- Main Host Code ---
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image.pgm> <output_image.pgm> \n";
        return 1;
    }

    std::string inputFilename = argv[1];
    std::string outputFilename = argv[2];

    // 1. Load the input image
    Image inputImage = loadImage(inputFilename);
    if (inputImage.channels != 1) {
        std::cerr << "Error: This naive convolution code expects a grayscale (P5) image. Please provide a .pgm file." << std::endl;
        return 1;
    }
    std::cout << "Loaded image: " << inputFilename << " (" << inputImage.width << "x" << inputImage.height << ", " << inputImage.channels << " channels)\n";

    // Prepare output image buffer
    Image outputImage = inputImage; // Copy metadata, data will be overwritten
    outputImage.data.assign(inputImage.data.size(), 0); // Initialize with zeros

    unsigned char* d_inputImage = nullptr;
    unsigned char* d_outputImage = nullptr;

    size_t imageSize = static_cast<size_t>(inputImage.width) * inputImage.height * inputImage.channels * sizeof(unsigned char);

    // 2. Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_inputImage, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_outputImage, imageSize));

    // 3. Copy input image from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_inputImage, inputImage.data.data(), imageSize, cudaMemcpyHostToDevice));

    // Define kernel launch parameters
    dim3 threadsPerBlock(32, 8);
    dim3 numBlocks(
        (inputImage.width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (inputImage.height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    std::cout << "Launching kernel with " << numBlocks.x << "x" << numBlocks.y << " blocks and "
              << threadsPerBlock.x << "x" << threadsPerBlock.y << " threads per block.\n";

    // Set up CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // 4. Launch the naive convolution kernel
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    naiveConvolution<<<numBlocks, threadsPerBlock>>>(
        d_inputImage,
        d_outputImage,
        inputImage.width,
        inputImage.height
    );
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for errors during kernel execution
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Naive kernel execution time: " << milliseconds << " ms\n";

    // 5. Copy output image from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(outputImage.data.data(), d_outputImage, imageSize, cudaMemcpyDeviceToHost));

    // 6. Save the convolved image
    saveImage(outputFilename, outputImage);
    std::cout << "Saved blurred image to: " << outputFilename << std::endl;

    // 7. Clean up device memory
    CHECK_CUDA_ERROR(cudaFree(d_inputImage));
    CHECK_CUDA_ERROR(cudaFree(d_outputImage));

    // Destroy events
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    std::cout << "Program finished successfully.\n";

    return 0;
}