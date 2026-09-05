#include "cudavision/convolution.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define TILE_WIDTH 16
#define FILTER_RADIUS 3
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))
#define TILE_DIM 32

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

__constant__ float F[FILTER_SIZE * FILTER_SIZE];
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

void cvgpu::uploadConvolutionFilter(const float *filter)
{
    CUDA_CHECK(cudaMemcpyToSymbol(F, filter, sizeof(F)));
    CUDA_CHECK(cudaMemcpyToSymbol(F_c, filter, sizeof(F_c)));
}

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

__global__ void convolution_2D_const_mem_kernel
(
    float *input,
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
                Pvalue += F[fRow * FILTER_SIZE + fCol] * input[inRow * width + inCol];
            }
        }
    }
    output[outRow*width + outCol] = Pvalue;
}

__global__ void convolution_tiled_2D_const_mem_kernel
(
    float *input,
    float *output,
    int width,
    int height
)
{
    int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    // loading input tile
    __shared__ float input_shared[IN_TILE_DIM][IN_TILE_DIM];
    if (row >= 0 && row<height && col >= 0 && col < width)
    {
        input_shared[threadIdx.y][threadIdx.x] = input[row*width + col];
    }
    else
    {
        input_shared[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // calculating output elements
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    // turning off the threads at the edges of the block
    if (col >= 0 && col < width && row >= 0 && row < height)
    {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >=0 && tileRow < OUT_TILE_DIM)
        {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2*FILTER_RADIUS + 1; fRow++)
            {
                for (int fCol = 0; fCol < 2*FILTER_RADIUS + 1; fCol++)
                {
                    Pvalue += F_c[fRow][fCol]*input_shared[tileRow+fRow][tileCol+fCol];
                }
            }
            output[row*width+col] = Pvalue;
        }
    }
}

__global__ void convolution_cached_tiled_2D_const_mem
(
    float *input,
    float *output,
    int width,
    int height
)
{
    int col = blockIdx.x*TILE_DIM + threadIdx.x;
    int row = blockIdx.y*TILE_DIM + threadIdx.y;

    // loading input tile
    __shared__ float input_shared[TILE_DIM][TILE_DIM];
    if (row >= 0 && row<height && col >= 0 && col < width)
    {
        input_shared[threadIdx.y][threadIdx.x] = input[row*width + col];
    }
    else
    {
        input_shared[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // calculating output elements
    // turning off the threads at the edges of the block
    if (col < width && row < height)
    {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++)
        {
            for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++)
            {
                if (threadIdx.x-FILTER_RADIUS+fCol >= 0 &&
                    threadIdx.x-FILTER_RADIUS+fCol < TILE_DIM &&
                    threadIdx.y-FILTER_RADIUS+fRow >= 0 &&
                    threadIdx.y-FILTER_RADIUS+fRow < TILE_DIM)
                {
                    Pvalue += F_c[fRow][fCol] * input_shared[
                        threadIdx.y - FILTER_RADIUS + fRow
                    ][
                        threadIdx.x - FILTER_RADIUS + fCol
                    ];
                }
                else
                {
                    if (row-FILTER_RADIUS+fRow >= 0 &&
                        row-FILTER_RADIUS+fRow < height &&
                        col-FILTER_RADIUS+fCol >= 0 &&
                        col-FILTER_RADIUS+fCol < width)
                    {
                        Pvalue += F_c[fRow][fCol]*input[(row-FILTER_RADIUS+fRow)*width+col-FILTER_RADIUS+fCol];
                    }
                }
            }
        }
        output[row*width+col] = Pvalue;
    }
}

void runConvolution
(
    float *input,
    float *filter,
    float *output,
    int radius,
    int width,
    int height,
    bool tiled,
    bool using_const,
    bool cached
)
{
    size_t size = static_cast<size_t>(width) * height * sizeof(float);
    size_t filter_size = static_cast<size_t>(2*radius+1) * (2*radius+1) * sizeof(float);

    // GPU allocation
    float* input_d;
    float* output_d;
    float* filter_d = nullptr;

    // GPU allocation
    CUDA_CHECK(cudaMalloc(&input_d, size));
    CUDA_CHECK(cudaMalloc(&output_d, size));

    // cudaMemcpy
    CUDA_CHECK(cudaMemcpy(input_d, input, size, cudaMemcpyHostToDevice));
    if (!tiled && !using_const)
    {
        CUDA_CHECK(cudaMalloc(&filter_d, filter_size));
        CUDA_CHECK(cudaMemcpy(filter_d, filter, filter_size, cudaMemcpyHostToDevice));
    }

    constexpr int BLOCK_SIZE = 16;
    // kernel launch
    dim3 blockDim;
    dim3 gridDim;

    if (cached)
    {
        blockDim = dim3(TILE_DIM, TILE_DIM);
        gridDim = dim3(
            (width + TILE_DIM - 1) / TILE_DIM,
            (height + TILE_DIM - 1) / TILE_DIM
        );
    }
    else if (tiled)
    {
        blockDim = dim3(IN_TILE_DIM, IN_TILE_DIM);
        gridDim = dim3(
            (width + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
            (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM
        );
    }
    else
    {
        blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE);
        gridDim = dim3(
            (width + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (height + BLOCK_SIZE - 1) / BLOCK_SIZE
        );
    }

    if (cached)
    {
        convolution_cached_tiled_2D_const_mem<<<gridDim, blockDim>>>(
            input_d, output_d, width, height
        );
    }
    else if (tiled)
    {
        convolution_tiled_2D_const_mem_kernel<<<gridDim, blockDim>>>(
            input_d, output_d, width, height
        );
    }
    else if (using_const)
    {
        convolution_2D_const_mem_kernel<<<gridDim, blockDim>>>(
            input_d, output_d, radius, width, height
        );
    }
    else
    {
        convolution_2D_basic_kernel<<<gridDim, blockDim>>>(
            input_d, filter_d, output_d, radius, width, height
        );
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
    if (filter_d != nullptr)
    {
        CUDA_CHECK(cudaFree(filter_d));
    }
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
    runConvolution(input, filter, output, radius, width, height, false, false, false);
}

void cvgpu::convolution_2D_const_mem(
    float *input,
    float *output,
    int radius,
    int width,
    int height
)
{
    runConvolution(input, nullptr, output, radius, width, height, false, true, false);
}

void cvgpu::convolution_tiled_2D_const_mem(
    float *input,
    float *output,
    int width,
    int height
)
{
    runConvolution(input, nullptr, output, FILTER_RADIUS, width, height, true, true, false);
}

void cvgpu::convolution_cached_tiled_2D_const_mem(
    float *input,
    float *output,
    int width,
    int height
)
{
    runConvolution(input, nullptr, output, FILTER_RADIUS, width, height, false, true, true);
}