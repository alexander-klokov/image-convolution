
// Define block and shared memory tile dimensions
// Tile dimensions include the padding needed for the filter

__global__ void kernelTilingPadding(
    unsigned char *d_input,
    unsigned char *d_output,
    int originalWidth,
    int originalHeight,
    int paddedWidth,
    const float filterValue)
{

    // Declare a shared memory tile for the input data + filter halo
    const int BOX_FILTER_SIZE = 41;
    const int HALF_BOX_FILTER = BOX_FILTER_SIZE / 2;

    const int BLOCK_WIDTH = 32;
    const int BLOCK_HEIGHT = 32;
    const int TILEWIDTH = BLOCK_WIDTH + 2 * HALF_BOX_FILTER;
    const int TILEHEIGHT = BLOCK_HEIGHT + 2 * HALF_BOX_FILTER;

    __shared__ float sh_tile[TILEHEIGHT][TILEWIDTH];

    // Calculate global thread coordinates
    int x = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
    int y = blockIdx.y * BLOCK_HEIGHT + threadIdx.y;

    // Calculate the coordinates of the tile's top-left corner in global memory
    int tileOriginX = x - threadIdx.x - HALF_BOX_FILTER;
    int tileOriginY = y - threadIdx.y - HALF_BOX_FILTER;

    const int loadLimit = TILEWIDTH * TILEHEIGHT;
    const int loadStride = BLOCK_WIDTH * BLOCK_HEIGHT;

    for (int i = threadIdx.y * BLOCK_WIDTH + threadIdx.x; i < loadLimit; i += loadStride)
    {

        int ty = i / TILEWIDTH;
        int tx = i % TILEWIDTH;

        int currentX = tileOriginX + tx;
        int currentY = tileOriginY + ty;

        // Check if the source pixel is within the original image bounds
        if (currentX >= 0 && currentX < originalWidth && currentY >= 0 && currentY < originalHeight)
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

    __syncthreads(); // Wait for all threads in the block to finish loading the tile

    // Exit if the thread is outside the original image bounds
    if (x >= originalWidth || y >= originalHeight)
    {
        return;
    }

    // --- Perform the convolution using shared memory ---
    float sum = 0.0f;
    for (int j = 0; j < BOX_FILTER_SIZE; ++j)
    {
        for (int i = 0; i < BOX_FILTER_SIZE; ++i)
        {

            sum += sh_tile[threadIdx.y + j][threadIdx.x + i];
        }
    }
    sum *= filterValue;

    d_output[y * paddedWidth + x] = static_cast<unsigned char>(min(255.0f, max(0.0f, sum)));
}