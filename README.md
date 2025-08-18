# CUDA Kernel Optimization for Image Convolution

## Motivation

Starting this study project, I was inspired by _Simon Boehm_ and his [post](https://siboehm.com/articles/22/CUDA-MMM), in which he was optimizing a CUDA Matmul Kernel trying to achieve cuBLAS performance.

I plan to develop a basic image convolution CUDA kernel and enhance it through iterative optimization techniques. For performance comparison, I will use the _nppiFilter_8u_C1R_ function from the NVIDIA Performance Primitives (NPP) library as a benchmark.

## Input image and the Convolution Kernel

As input, I am taking a PGM image. A PGM image (Portable Gray Map) is a straightforward file format for storing 2D grayscale images, with each pixel representing a shade of gray. The single channel of the image simplifies the problem.

I’m using a straightforward yet extended convolution kernel - a 41x41 box filter. That makes the convolution computationally intense, offering significant room for optimization, and produces a pleasantly blurred output image.

I'm working locally using NVIDIA GeForce RTX 4060 Laptop GPU.

## Benchmark: nppiFilter_8u_C1R

_nppiFilter_8u_C1R_ is a CPU function, not a GPU kernel. When invoked, it does the following:

- prepares data structures and parameters, including calculating the optimal _gridSize_ and _blockSize_;
- launches pre-compiled kernels optimized for the input image and the specific GPU architecture;
- Handles synchronization and data flow.

The automatically selected kernel _ForEachPixelNaiveInLargeImage_ demonstrated nearly perfect performance:

- Compute Throughput (%): 99.68
- Memory Throughput (%): 99.68
- Duration (ms): 93

It's interesting to note the parameters chosen:

- Grid Size: (125, 373, 1)
- Block Size: (32, 8, 1)

## Kernel 1a: Naive Implementation

In this straightforward approach, each thread processes a specific pixel in the output image. I've set the block size to 32 and, then, determined the grid size by overlaying the input image with these blocks.

The kernel is not performing enough arithmetic work:

- Compute Throughput (%): 16.56
- Memory Throughput (%): 90.26
- Duration (ms): 712

with

- Grid Size: (126, 95, 1)
- Block Size: (32, 32, 1)

## Kernel 1b: Naive Implementation with Optimal Run Parameters

Before getting to any optimization, I ran the same naive kernel using the run parameters recommended by the NVIDIA Performance Primitives (NPP) function. Only a small improvement was achieved:

- Compute Throughput (%): 17.20
- Memory Throughput (%): 91.32
- Duration (ms): 682

with

- Grid Size: (126, 378, 1)
- Block Size: (32, 8, 1)


