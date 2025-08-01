# CUDA Kernel Optimization for Image Convolution

## Motivation

Starting this study project, I was inspired by _Simon Boehm_ and his [post](https://siboehm.com/articles/22/CUDA-MMM), in which he was optimizing a CUDA Matmul Kernel trying to achieve cuBLAS performance.

I plan to develop a basic image convolution CUDA kernel and enhance it through iterative optimization techniques. For performance comparison, I will use the _nppiFilter_8u_C1R_ function from the NVIDIA Performance Primitives (NPP) library as a benchmark.

## Input image and the Convolution Kernel

As input, I am taking a PGM image. A PGM image (Portable Gray Map) is a straightforward file format for storing 2D grayscale images, with each pixel representing a shade of gray. The single channel of the image simplifies the problem.

I’m using a straightforward yet extended convolution kernel - a 41x41 box filter. That makes the convolution computationally intense, offering significant room for optimization, and produces a pleasantly blurred output image.

## Kernel 1: Naive Implementation

In this straightforward approach, each thread processes a specific pixel in the output image.



