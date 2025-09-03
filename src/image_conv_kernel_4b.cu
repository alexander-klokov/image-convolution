#include <iostream>
#include <chrono>
#include <algorithm>

#include <cuda_runtime.h>

#include "image_utils.h"
#include "kernels.cuh"

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_image.pgm> <output_image.pgm> \n";
        return 1;
    }

    std::string inputFilename = argv[1];
    std::string outputFilename = argv[2];

    std::cout << argv[1] << " " << argv[2] << std::endl;

    Image inputImage = loadImage(inputFilename);
    if (inputImage.channels != 1)
    {
        std::cerr << "Error: This convolution code expects a grayscale (P5) image. Please provide a .pgm file." << std::endl;
        return 1;
    }
    std::cout << "Loaded image: " << inputFilename << " (" << inputImage.width << "x" << inputImage.height << ", " << inputImage.channels << " channels)\n";

    // Prepare output image buffer
    Image outputImage = inputImage;                     // Copy metadata, data will be overwritten
    outputImage.data.assign(inputImage.data.size(), 0); // Initialize with zeros

    unsigned char *d_inputImage = nullptr;
    unsigned char *d_outputImage = nullptr;

    size_t imageSize = static_cast<size_t>(inputImage.width) * inputImage.height * inputImage.channels * sizeof(unsigned char);

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_inputImage, imageSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_outputImage, imageSize));

    // Copy input image from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_inputImage, inputImage.data.data(), imageSize, cudaMemcpyHostToDevice));

    // Filter parameters
    const short filterSize = 41;
    const short filterSize2 = filterSize * filterSize;
    const float filterValue = 1.0f / filterSize2;

    // Define kernel launch parameters
    const int BLOCK_WIDTH = 32;
    const int BLOCK_HEIGHT = 16;
    dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 gridDim(
        (inputImage.width + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
        (inputImage.height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);

    const int PADDING_ALIGNMENT = 32; // Or 32 for warp size
    int paddedWidth = ((inputImage.width + PADDING_ALIGNMENT - 1) / PADDING_ALIGNMENT) * PADDING_ALIGNMENT;
    int paddedHeight = ((inputImage.height + PADDING_ALIGNMENT - 1) / PADDING_ALIGNMENT) * PADDING_ALIGNMENT;
    size_t paddedImageSize = paddedWidth * paddedHeight * sizeof(unsigned char);

    // Allocate padded host memory
    unsigned char *h_paddedInputImage;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_paddedInputImage, paddedImageSize));

    // Copy original image data into padded host memory
    // This is an example, you need to implement this copy carefully
    for (int y = 0; y < inputImage.height; ++y)
    {
        memcpy(h_paddedInputImage + y * paddedWidth,
               inputImage.data.data() + y * inputImage.width,
               inputImage.width * sizeof(unsigned char));
    }

    // Allocate device memory with padding
    float *d_paddedInputImage;
    CHECK_CUDA_ERROR(cudaMalloc(&d_paddedInputImage, paddedImageSize));

    // Copy padded host data to padded device buffer
    CHECK_CUDA_ERROR(cudaMemcpy(d_paddedInputImage, h_paddedInputImage, paddedImageSize, cudaMemcpyHostToDevice));

    std::cout << "Launching kernelTilingPaddingHalf with " << gridDim.x << "x" << gridDim.y << " blocks and "
              << blockDim.x << "x" << blockDim.y << " threads per block.\n";

    // Set up CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Launch the kernel
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    kernelTilingPaddingHalf<<<gridDim, blockDim>>>(
        d_inputImage,
        d_outputImage,
        inputImage.width,
        inputImage.height,
        paddedWidth,
        filterValue);

    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for errors during kernel execution
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "\033[1;34mkernelTilingL2Cache execution time: " << milliseconds << " ms\n\033[0m";

    // Copy output image from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(outputImage.data.data(), d_outputImage, imageSize, cudaMemcpyDeviceToHost));

    // Save the convolved image
    saveImage(outputFilename, outputImage);
    std::cout << "Saved blurred image to: " << outputFilename << std::endl;

    // Clean up device memory
    CHECK_CUDA_ERROR(cudaFree(d_inputImage));
    CHECK_CUDA_ERROR(cudaFree(d_outputImage));

    // Destroy events
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    std::cout << "\033[1;32mCompleted...\033[0m\n";

    return 0;
}
