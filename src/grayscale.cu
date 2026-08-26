#include "cudavision/grayscale.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CHANNELS 3
#define TILE_WIDTH 16

#define CUDA_CHECK(call)                                             \
    do {                                                             \
        cudaError_t error = (call);                                  \
        if (error != cudaSuccess) {                                  \
            std::cerr << "CUDA error: "                              \
                      << cudaGetErrorString(error)                   \
                      << " at " << __FILE__                          \
                      << ":" << __LINE__ << '\n';                    \
            std::exit(EXIT_FAILURE);                                 \
        }                                                            \
    } while (0)

__global__ void grayscaleKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
) 
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    bool inBounds = col < width && row < height;

    if (inBounds) {
        int grayOffset = row*width+col;
        int rgbOffset = grayOffset*CHANNELS;

        unsigned char r = input[rgbOffset];
        unsigned char g = input[rgbOffset+1];
        unsigned char b = input[rgbOffset+2];

        output[grayOffset] = (unsigned char) (0.21f*r+0.71f*g+0.07*b);
    }
}

// tiled implementation
__global__ void tiledGrayscaleKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
) 
{
    __shared__ unsigned char tile[TILE_WIDTH][TILE_WIDTH][CHANNELS];

    int col = blockIdx.x*TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y*TILE_WIDTH + threadIdx.y;
    bool inBounds = col < width && row < height;
    int grayOffset = row*width+col;

    if (inBounds)
    {
        int rgbOffset = grayOffset*CHANNELS;

        // load RGB from global memory
        unsigned char r = input[rgbOffset];
        unsigned char g = input[rgbOffset+1];
        unsigned char b = input[rgbOffset+2];

        // store RGB in shared memory
        tile[threadIdx.y][threadIdx.x][0] = r;
        tile[threadIdx.y][threadIdx.x][1] = g;
        tile[threadIdx.y][threadIdx.x][2] = b;
    } else 
    {
        tile[threadIdx.y][threadIdx.x][0] = 0;
        tile[threadIdx.y][threadIdx.x][1] = 0;
        tile[threadIdx.y][threadIdx.x][2] = 0;
    }

    // All threads must reach the barrier, including out-of-bounds threads.
    __syncthreads();

    if (inBounds) {
        // read RGB from shared memory
        unsigned char R = tile[threadIdx.y][threadIdx.x][0];
        unsigned char G = tile[threadIdx.y][threadIdx.x][1];
        unsigned char B = tile[threadIdx.y][threadIdx.x][2];
        
        // calculate grayscale
        unsigned char pixel = (unsigned char) (0.21f*R+0.71f*G+0.07*B);
        
        output[grayOffset] = pixel;
    }
}

void runGrayscale(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    bool tiled
) 
{
    size_t rgbSize  = width * height * CHANNELS * sizeof(unsigned char);
    size_t graySize = width * height * sizeof(unsigned char);
    unsigned char *input_device, *output_device;
    
    // GPU allocation
    CUDA_CHECK(cudaMalloc((void**) &input_device, rgbSize));
    CUDA_CHECK(cudaMalloc((void**) &output_device, graySize));

    // cudaMemcpy
    CUDA_CHECK(cudaMemcpy(input_device, input, rgbSize, cudaMemcpyHostToDevice));

    // kernel launch
    constexpr int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
        (height + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    if (tiled) {
        tiledGrayscaleKernel<<<gridDim, blockDim>>>(
            input_device, output_device, width, height
        );
    } else {
        grayscaleKernel<<<gridDim, blockDim>>>(
            input_device, output_device, width, height
        );
    }

    // Check whether the kernel launch itself was valid
    CUDA_CHECK(cudaGetLastError());

    // Wait for the kernel and catch execution errors
    CUDA_CHECK(cudaDeviceSynchronize());

    // cudaMemcpy back
    CUDA_CHECK(cudaMemcpy(output, output_device, graySize, cudaMemcpyDeviceToHost));

    // free memory
    CUDA_CHECK(cudaFree(input_device));
    CUDA_CHECK(cudaFree(output_device));
}

void cvgpu::grayscale(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
)
{
    runGrayscale(input, output, width, height, false);
}

void cvgpu::grayscaleTiled(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
)
{
    runGrayscale(input, output, width, height, true);
}