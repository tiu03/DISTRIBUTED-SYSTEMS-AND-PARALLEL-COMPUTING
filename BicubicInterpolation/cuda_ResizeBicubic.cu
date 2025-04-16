#include <cuda_runtime.h>
#include <iostream>
#include "serial_ResizeBicubic.h"

using namespace std;

// for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " : " << cudaGetErrorString(err) << endl; \
            exit(err); \
        } \
    } while (0)

// Refer to bicubicKernal in bicubicKernel.cpp
__device__ float cuda_bicubicKernel(float d) {
    d = fabsf(d);
    if (d <= 1.0f) {
        return (1.5f * d * d * d - 2.5f * d * d + 1.0f);
    }
    else if (d <= 2.0f) {
        return (-0.5f * d * d * d + 2.5f * d * d - 4.0f * d + 2.0f);
    }
    return 0.0f;
}

// Refer to cuda_getPixelValue in bicubicKernel.cpp
__device__ float cuda_getPixelValue(unsigned char* image, int width, int height, int channels, int x, int y, int c) {
    x = max(0, min(x, width - 1));
    y = max(0, min(y, height - 1));
    return image[(y * width + x) * channels + c];
}

// CUDA kernel for bicubic resizing
__global__ void cuda_ResizeBicubicKernel(unsigned char* src, int srcWidth, int srcHeight, int channels,
    unsigned char* dst, int dstWidth, int dstHeight, float scaleX, float scaleY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dstWidth && y < dstHeight) {
        for (int c = 0; c < channels; ++c) {
            float srcX = x * scaleX;
            float srcY = y * scaleY;
            int x1 = (int)srcX;
            int y1 = (int)srcY;

            float result = 0.0f;
            for (int m = -1; m <= 2; ++m) {
                for (int n = -1; n <= 2; ++n) {
                    float weight = cuda_bicubicKernel(srcX - (x1 + n)) * cuda_bicubicKernel(srcY - (y1 + m));
                    result += cuda_getPixelValue(src, srcWidth, srcHeight, channels, x1 + n, y1 + m, c) * weight;
                }
            }

            dst[(y * dstWidth + x) * channels + c] = min(max((int)result, 0), 255);
        }
    }
}

// Function to resize the image on the GPU
void cuda_ResizeBicubic(unsigned char* src, int srcWidth, int srcHeight, int channels,
    unsigned char* dst, int dstWidth, int dstHeight) {
    unsigned char* d_src;
    unsigned char* d_dst;
    size_t srcSize = srcWidth * srcHeight * channels * sizeof(unsigned char);
    size_t dstSize = dstWidth * dstHeight * channels * sizeof(unsigned char);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_src, srcSize));
    CUDA_CHECK(cudaMalloc((void**)&d_dst, dstSize));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_src, src, srcSize, cudaMemcpyHostToDevice));

    float scaleX = (float)srcWidth / dstWidth;
    float scaleY = (float)srcHeight / dstHeight;

    // Define CUDA block and grid sizes
    dim3 blockSize(32, 32);  // 16x16 threads per block
    dim3 gridSize((dstWidth + blockSize.x - 1) / blockSize.x, (dstHeight + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    cuda_ResizeBicubicKernel << <gridSize, blockSize >> > (d_src, srcWidth, srcHeight, channels, d_dst, dstWidth, dstHeight, scaleX, scaleY);
    cudaDeviceSynchronize();  // Ensure the kernel finishes before moving on


    // Copy the result back to host
    CUDA_CHECK(cudaMemcpy(dst, d_dst, dstSize, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}
