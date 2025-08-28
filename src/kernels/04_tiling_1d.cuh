#include <cuda_runtime.h>

#include <stdio.h>

#define N 41

#define FILTER_RADIUS (N / 2)

#define TILE_WIDTH 32

__global__ void kernelTiling1D(
    unsigned char *d_inputImage,
    unsigned char *d_outputImage,
    int width,
    int height,
    const float filterValue)
{
    const int sTileWidth = TILE_WIDTH + N - 1;

    __shared__ unsigned char s_tile[sTileWidth][sTileWidth];
    
    const int loadLimit = sTileWidth * sTileWidth;
    const int loadStride = blockDim.x * blockDim.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gx = blockIdx.x * TILE_WIDTH + tx;
    int gy = blockIdx.y * TILE_WIDTH + ty;

    // Load Phase: Reorganized for better memory coalescing
    for (int load_idx = tx + ty * blockDim.x; load_idx < loadLimit; load_idx += loadStride)
    {
        int load_x = load_idx % sTileWidth;
        int load_y = load_idx / sTileWidth;

        // Calculate global coordinates for the pixel to load
        int input_x = blockIdx.x * TILE_WIDTH - FILTER_RADIUS + load_x;
        int input_y = blockIdx.y * TILE_WIDTH - FILTER_RADIUS + load_y;

        if (input_x >= 0 && input_x < width && input_y >= 0 && input_y < height)
        {
            s_tile[load_y][load_x] = d_inputImage[input_y * width + input_x];
        }
        else
        {
            s_tile[load_y][load_x] = 0;
        }
    }

    // Sync threads
    __syncthreads();

    // Compute Phase: Using shared memory
    if (gx < width && gy < height)
    {
        float sum = 0.0f;
        for (int kRow = 0; kRow < N; ++kRow)
        {
            for (int kCol = 0; kCol < N; ++kCol)
            {
                sum += s_tile[ty + kRow][tx + kCol];
            }
        }

        sum *= filterValue;

        d_outputImage[gy * width + gx] = static_cast<unsigned char>(min(255.0f, max(0.0f, sum)));
    }

    return;
}