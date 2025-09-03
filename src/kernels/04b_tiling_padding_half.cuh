#include <cuda_runtime.h>
#include <stdio.h>

#define N 41
#define FILTER_RADIUS (N / 2)

__global__ void kernelTilingPaddingHalf(
    unsigned char *d_input,
    unsigned char *d_output,
    const int width,
    const int height,
    const int paddedWidth,
    const float filterValue)
{

    const int BLOCK_WIDTH = 32;
    const int BLOCK_HEIGHT = 16;

    // Global thread coordinates
    int x = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
    int y = blockIdx.y * BLOCK_HEIGHT + threadIdx.y;

    // Exit if the thread is outside the original image bounds
    if (x >= width || y >= height)
    {
        return;
    }

    const int TILEWIDTH = BLOCK_WIDTH + 2 * FILTER_RADIUS;
    const int TILEHEIGHT = BLOCK_HEIGHT + 2 * FILTER_RADIUS;

    __shared__ float sh_tile[TILEHEIGHT][TILEWIDTH];

    // Calculate the coordinates of the tile's top-left corner in global memory
    int tileOriginX = x - threadIdx.x - FILTER_RADIUS;
    int tileOriginY = y - threadIdx.y - FILTER_RADIUS;

    // Load Phase: Coalesced read
    const int loadLimit = TILEWIDTH * TILEHEIGHT;
    const int loadStride = BLOCK_WIDTH * BLOCK_HEIGHT;

    for (int i = threadIdx.y * BLOCK_WIDTH + threadIdx.x; i < loadLimit; i += loadStride)
    {

        int ty = i / TILEWIDTH;
        int tx = i % TILEWIDTH;

        int currentX = tileOriginX + tx;
        int currentY = tileOriginY + ty;

        // Check if the source pixel is within the original image bounds
        if (currentX >= 0 && currentX < width && currentY >= 0 && currentY < height)
        {
            // Read from global memory using the padded width
            sh_tile[ty][tx] = d_input[currentY * paddedWidth + currentX];
        }
        else
        {
            // Set boundary pixels to 0 or another suitable value for the convolution
            sh_tile[ty][tx] = 0.0f;
        }
    }

    // Sync threads
    __syncthreads();

    // Compute Phase: Using shared memory
    float sum = 0.0f;
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < N; ++i)
        {
            sum += sh_tile[threadIdx.y + j][threadIdx.x + i];
        }
    }
    sum *= filterValue;

    d_output[y * paddedWidth + x] = static_cast<unsigned char>(min(255.0f, max(0.0f, sum)));

    return;
}