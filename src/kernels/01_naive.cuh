#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// --- Naive CUDA Kernel ---

// Implements a 31x31 Box Blur convolution
__global__ void naiveConvolution(unsigned char* d_inputImage, unsigned char* d_outputImage, int width, int height) {
    // 31x31 Box Blur Kernel (all elements 1/1681)
    // Declared directly in kernel for simplicity in naive version.
    // In optimized version, this would be in constant memory or passed.
    const short kernelSize = 41;
    const short kernelSize2 = kernelSize * kernelSize;

    float kernel[kernelSize2];
    for (int i = 0; i < kernelSize2; ++i) {
        kernel[i] = 1.0f / kernelSize2;
    }
    int kernelRadius = kernelSize / 2;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the current thread is within image bounds
    // if (col < width && row < height) {
    if (col >= width || row >= height) {
        return;
    }

    float sum = 0.0f;

    // Iterate over the kernel window
    for (int kRow = -kernelRadius; kRow <= kernelRadius; ++kRow) {
        for (int kCol = -kernelRadius; kCol <= kernelRadius; ++kCol) {
            int inputRow = row + kRow;
            int inputCol = col + kCol;

            // Handle boundary conditions: Clamp to edge
            // This means pixels outside the image boundary are treated as if they are the nearest edge pixel.
            inputRow = min(max(inputRow, 0), height - 1);
            inputCol = min(max(inputCol, 0), width - 1);

            // Calculate kernel index
            int kIdxRow = kRow + kernelRadius;
            int kIdxCol = kCol + kernelRadius;

            // Accumulate weighted sum
            sum += d_inputImage[inputRow * width + inputCol] * kernel[kIdxRow * kernelSize + kIdxCol];
        }
    }
    // Write the result to the output image, clamping to 255 if sum exceeds
    d_outputImage[row * width + col] = static_cast<unsigned char>(min(255.0f, max(0.0f, sum)));
}