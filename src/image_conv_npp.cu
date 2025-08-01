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

    std::string inputFilename = argv[1];
    std::string outputFilename = argv[2];
    
    // input image
    Image inputImage = loadImage(inputFilename);
    const short image_width = inputImage.width;
    const short image_height = inputImage.height;
    
    // convolution kernel
    const int kernel_size = 41;
    const int kernel_elements = kernel_size * kernel_size;
    std::vector<Npp32s> hKernel(kernel_elements, 1); // a simple box filter
    NppiPoint oAnchor = {kernel_size / 2, kernel_size / 2}; // center of the kernel
    int divisor = kernel_elements; // for a box filter, the divisor is the sum of elements.

    // device memory allocation
    Npp8u* dSrc = nullptr;
    Npp8u* dDst = nullptr;
    Npp32s* dKernel = nullptr;
    size_t dSrcStep, dDstStep;

    // -- source image
    checkCudaErrors(cudaMallocPitch(&dSrc, &dSrcStep, image_width * sizeof(Npp8u), image_height));
    
    // -- destination image
    NppiSize oKernelSize = {kernel_size, kernel_size};
    NppiSize oSizeROI = {image_width - kernel_size + 1, image_height - kernel_size + 1};

    checkCudaErrors(cudaMallocPitch(&dDst, &dDstStep, oSizeROI.width * sizeof(Npp8u), oSizeROI.height));

    // -- kernel
    checkCudaErrors(cudaMalloc(&dKernel, kernel_elements * sizeof(Npp32s)));

    // transfer data: H2D
    // -- source image
    checkCudaErrors(cudaMemcpy2D(dSrc, dSrcStep, inputImage.data.data(), image_width * sizeof(Npp8u),
                                 image_width * sizeof(Npp8u), image_height, cudaMemcpyHostToDevice));    
    // -- kernel
    checkCudaErrors(cudaMemcpy(dKernel, hKernel.data(), kernel_elements * sizeof(Npp32s), cudaMemcpyHostToDevice));

    // convolution
    NppStatus nppStatus = nppiFilter_8u_C1R(dSrc, dSrcStep, dDst, dDstStep, 
                                             oSizeROI, dKernel, oKernelSize, 
                                             oAnchor, divisor);
    checkCudaErrors(cudaDeviceSynchronize());

    // transfer data: D2H
    std::vector<Npp8u> hDst(oSizeROI.width * oSizeROI.height);
    
    checkCudaErrors(cudaMemcpy2D(hDst.data(), oSizeROI.width * sizeof(Npp8u), dDst, dDstStep,
                                 oSizeROI.width * sizeof(Npp8u), oSizeROI.height, cudaMemcpyDeviceToHost));
    
    // save the filtered image
    save_pgm(hDst, oSizeROI.width, oSizeROI.height, outputFilename);

    // clean up device memory
    cudaFree(dSrc);
    cudaFree(dDst);
    cudaFree(dKernel);

    return 0;
}