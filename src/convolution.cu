#include "cudavision/convolution.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

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

__global__ void convolution_2D_basic_kernel
(
    float *input,
    float *filter,
    float *output,
    int radius,
    int width,
    int height
)
{
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;

    if (outCol >= width || outRow >= height) return; 

    float Pvalue = 0.0f;

    for (int fRow = 0; fRow < 2*radius+1; fRow++)
    {
        for (int fCol = 0; fCol < 2*radius+1; fCol++)
        {
            int inRow = outRow - radius + fRow;
            int inCol = outCol - radius + fCol;

            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
            {
                Pvalue += filter[fRow*(2*radius+1) + fCol] * input[inRow*width + inCol];
            }
        }
    }
    output[outRow*width + outCol] = Pvalue;
}

void runConvolution
(
    float *input,
    float *filter,
    float *output,
    int radius,
    int width,
    int height,
    bool tiled
)
{
    size_t size = static_cast<size_t>(width) * height * sizeof(float);
    size_t filter_size = static_cast<size_t>(2*radius+1) * (2*radius+1) * sizeof(float);

    // GPU allocation
    float* input_d;
    float* output_d;
    float* filter_d;
    
    // GPU allocation
    CUDA_CHECK(cudaMalloc(&input_d, size));
    CUDA_CHECK(cudaMalloc(&output_d, size));
    CUDA_CHECK(cudaMalloc(&filter_d, filter_size));

    // cudaMemcpy
    CUDA_CHECK(cudaMemcpy(input_d, input, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(filter_d, filter, filter_size, cudaMemcpyHostToDevice));


    // kernel launch
    constexpr int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
        (height + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    
    if (tiled)
    {
        // (tiled version not implemented yet – clean up and return)
        CUDA_CHECK(cudaFree(input_d));
        CUDA_CHECK(cudaFree(output_d));
        CUDA_CHECK(cudaFree(filter_d));
        return;
    }
    else
    {
        convolution_2D_basic_kernel<<<gridDim, blockDim>>>(input_d, filter_d, output_d, radius, width, height);
    }
    

    // Check whether the kernel launch itself was valid
    CUDA_CHECK(cudaGetLastError());

    // Wait for the kernel and catch execution errors
    CUDA_CHECK(cudaDeviceSynchronize());

    // cudaMemcpy back
    CUDA_CHECK(cudaMemcpy(output, output_d, size, cudaMemcpyDeviceToHost));

    // free memory
    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(output_d));
    CUDA_CHECK(cudaFree(filter_d));
}

void cvgpu::convolution_2D_basic(
    float *input,
    float *filter,
    float *output,
    int radius,
    int width,
    int height
)
{
    runConvolution(input, filter, output, radius, width, height, false);
}

