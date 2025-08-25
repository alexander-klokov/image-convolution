#pragma once

#include <cstdio>
#include <cstdlib>
// #include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void kernelNaive(unsigned char *d_inputImage, unsigned char *d_outputImage, int width, int height)
{
    // 41x41 Box Blur Kernel (all elements 1/1681).
    // Declared directly in kernel for simplicity in naive version.
    // In optimized version, this would be in constant memory or passed.
    const short filterSize = 41;
    const short filterSize2 = filterSize * filterSize;

    float kernel[filterSize2];
    for (int i = 0; i < filterSize2; ++i)
    {
        kernel[i] = 1.0f / filterSize2;
    }
    int filterRadius = filterSize / 2;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the current thread is within image bounds
    if (col >= width || row >= height)
    {
        return;
    }

    float sum = 0.0f;

    // Iterate over the kernel window
    for (int kRow = -filterRadius; kRow <= filterRadius; ++kRow)
    {
        for (int kCol = -filterRadius; kCol <= filterRadius; ++kCol)
        {
            int inputRow = row + kRow;
            int inputCol = col + kCol;

            // Handle boundary conditions: Clamp to edge
            // This means pixels outside the image boundary are treated as if they are the nearest edge pixel.
            inputRow = min(max(inputRow, 0), height - 1);
            inputCol = min(max(inputCol, 0), width - 1);

            // Calculate kernel index
            int kIdxRow = kRow + filterRadius;
            int kIdxCol = kCol + filterRadius;

            // Accumulate weighted sum
            sum += d_inputImage[inputRow * width + inputCol] * kernel[kIdxRow * filterSize + kIdxCol];
        }
    }
    // Write the result to the output image, clamping to 255 if sum exceeds
    d_outputImage[row * width + col] = static_cast<unsigned char>(min(255.0f, max(0.0f, sum)));

    return;
}