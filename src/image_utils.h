#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <string>
#include <vector>

#include <cuda_runtime.h>

// cuda
void check(cudaError_t err, const char *const func, const char *const file, const int line);

// image
struct Image
{
    std::vector<unsigned char> data;
    int width;
    int height;
    int channels; // 1 for grayscale, 3 for RGB
};

Image loadImage(const std::string &filename);

void saveImage(const std::string &filename, const Image &img);

#endif