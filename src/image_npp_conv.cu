#include <iostream>
#include <vector>
#include <numeric>
#include <fstream>

#include <npp.h>
#include <nppi.h>
#include <cuda_runtime.h>

#include "image_utils.h"

// Helper function to check for CUDA errors
void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void save_pgm(const std::vector<Npp8u>& image, int width, int height, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    // Write PGM header (P5 format)
    file << "P5\n" << width << " " << height << "\n255\n";

    // Write image data
    file.write(reinterpret_cast<const char*>(image.data()), width * height);

    file.close();
    std::cout << "Image successfully written to " << filename << std::endl;
}


int main(int argc, char** argv) {
    // --- 1. Define Image and Kernel Parameters ---
    const int image_width = 4032;
    const int image_height = 3024;
    const int kernel_size = 41;
    const int kernel_elements = kernel_size * kernel_size;

    // --- 2. Host Memory Allocation and Initialization ---
    std::string inputFilename = argv[1];
    Image inputImage = loadImage(inputFilename);

    // Convolution Kernel (host)
    std::vector<Npp32s> hKernel(kernel_elements, 1); // A simple box filter
    int divisor = kernel_elements; // For a box filter, the divisor is the sum of elements.

    // --- 3. Device Memory Allocation ---
    Npp8u* dSrc = nullptr;
    Npp8u* dDst = nullptr;
    Npp32s* dKernel = nullptr;
    size_t dSrcStep, dDstStep;

    // Allocate source image memory with pitch for optimal access
    checkCudaErrors(cudaMallocPitch(&dSrc, &dSrcStep, image_width * sizeof(Npp8u), image_height));
    
    // Calculate the size of the destination image
    NppiSize oKernelSize = {kernel_size, kernel_size};
    NppiSize oSizeROI = {image_width - kernel_size + 1, image_height - kernel_size + 1};

    // Allocate destination image memory with pitch
    checkCudaErrors(cudaMallocPitch(&dDst, &dDstStep, oSizeROI.width * sizeof(Npp8u), oSizeROI.height));

    // Allocate kernel memory on the device
    checkCudaErrors(cudaMalloc(&dKernel, kernel_elements * sizeof(Npp32s)));

    // --- 4. Copy Host Data to Device ---
    // Copy source image to device
    checkCudaErrors(cudaMemcpy2D(dSrc, dSrcStep, inputImage.data.data(), image_width * sizeof(Npp8u),
                                 image_width * sizeof(Npp8u), image_height, cudaMemcpyHostToDevice));
    
    // Copy kernel to device
    checkCudaErrors(cudaMemcpy(dKernel, hKernel.data(), kernel_elements * sizeof(Npp32s), cudaMemcpyHostToDevice));

    // --- 5. Perform the Convolution ---
    // Define the anchor point (center of the kernel)
    NppiPoint oAnchor = {kernel_size / 2, kernel_size / 2};

    std::cout << "Starting 41x41 convolution on a 4032x3024 image..." << std::endl;

    // Call the NPP function
    NppStatus nppStatus = nppiFilter_8u_C1R(dSrc, dSrcStep, dDst, dDstStep, 
                                             oSizeROI, dKernel, oKernelSize, 
                                             oAnchor, divisor);

    // Synchronize the device to ensure the operation is complete
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "Convolution complete." << std::endl;

    // --- 6. Copy Result Back to Host and Clean Up ---
    // Allocate host memory for the result
    std::vector<Npp8u> hDst(oSizeROI.width * oSizeROI.height);
    
    // Copy the result from device to host
    checkCudaErrors(cudaMemcpy2D(hDst.data(), oSizeROI.width * sizeof(Npp8u), dDst, dDstStep,
                                 oSizeROI.width * sizeof(Npp8u), oSizeROI.height, cudaMemcpyDeviceToHost));
    
    save_pgm(hDst, oSizeROI.width, oSizeROI.height, "output.pgm");


    

    // Clean up device memory
    cudaFree(dSrc);
    cudaFree(dDst);
    cudaFree(dKernel);

    std::cout << "Result copied to host memory and device memory freed." << std::endl;
    // You can now process or save the hDst vector.

    return 0;
}