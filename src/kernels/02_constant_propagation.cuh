#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void kernelConstantPropagation (
    unsigned char* d_inputImage,
    unsigned char* d_outputImage,
    int width,
    int height,
    const int filterRadius,
    const float filterValue
) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // check if the current thread is within image bounds
    if (col >= width || row >= height) {
        return;
    }

    float sum = 0.0f;

    // iterate over the filter window
    for (int kRow = -filterRadius; kRow <= filterRadius; ++kRow) {
        for (int kCol = -filterRadius; kCol <= filterRadius; ++kCol) {
            int inputRow = row + kRow;
            int inputCol = col + kCol;

            // handle boundary conditions: clamp to edge;
            // this means pixels outside the image boundary are treated as if they are the nearest edge pixel.
            inputRow = min(max(inputRow, 0), height - 1);
            inputCol = min(max(inputCol, 0), width - 1);

            // accumulate sum
            sum += d_inputImage[inputRow * width + inputCol];
        }
    }

    // weight the sum once the 'filterValue' is constant
    sum *= filterValue;

    // write the result to the output image, clamping to 255 if sum exceeds
    d_outputImage[row * width + col] = static_cast<unsigned char>(min(255.0f, max(0.0f, sum)));

    return;
}