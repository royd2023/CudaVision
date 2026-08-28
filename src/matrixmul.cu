#include "cudavision/matrixmul.hpp"
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

__global__ void matrixMulKernel(
    float* M,
    float* N,
    float* P,
    int width
) 
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    bool inBounds = col < width && row < width;

    if (inBounds) 
    {
        float Pvalue = 0;
        for (int k = 0; k < width; ++k)
        {
            Pvalue += M[row*width+k]*N[k*width+col];
        }
        P[row*width+col] = Pvalue;
    }
}

// tiled implementation
__global__ void tiledMatrixMulKernel(
    float* M,
    float* N,
    float* P,
    int width
) 
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // working on P[i,j]
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;
    // ph meaning phase
    for (int ph = 0; ph < width/TILE_WIDTH; ++ph)
    {   
        // load tiles into shared memory
        if ((row < width) && (ph*TILE_WIDTH+tx) < width)
        {
            Mds[ty][tx] = M[row*width + ph*TILE_WIDTH + tx];
        }
        else
        {
            Mds[ty][tx] = 0.0f;
        }

        if ((ph*TILE_WIDTH+ty) < width && col < width)
        {
            Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*width + col];
        }
        else
        {
            Nds[ty][tx] = 0.0f;
        }

        __syncthreads();

        // dot product in shared memory
        for (int k = 0; k < TILE_WIDTH; ++k)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    // store result in P
    P[row*width + col] = Pvalue;
}

void runMatrixMul(
    float* M,
    float* N,
    float* P,
    int width,
    bool tiled
) 
{
    size_t size = static_cast<size_t>(width) * width * sizeof(float);
    float* M_device;
    float* N_device;
    float* P_device;
    
    // GPU allocation
    CUDA_CHECK(cudaMalloc(&M_device, size));
    CUDA_CHECK(cudaMalloc(&N_device, size));
    CUDA_CHECK(cudaMalloc(&P_device, size));

    // cudaMemcpy
    CUDA_CHECK(cudaMemcpy(M_device, M, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_device, N, size, cudaMemcpyHostToDevice));


    // kernel launch
    constexpr int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
        (width + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    
    if (tiled)
    {
        tiledMatrixMulKernel<<<gridDim, blockDim>>>(M_device, N_device, P_device, width);
    }
    else
    {
        matrixMulKernel<<<gridDim, blockDim>>>(M_device, N_device, P_device, width);
    }
    

    // Check whether the kernel launch itself was valid
    CUDA_CHECK(cudaGetLastError());

    // Wait for the kernel and catch execution errors
    CUDA_CHECK(cudaDeviceSynchronize());

    // cudaMemcpy back
    CUDA_CHECK(cudaMemcpy(P, P_device, size, cudaMemcpyDeviceToHost));

    // free memory
    CUDA_CHECK(cudaFree(M_device));
    CUDA_CHECK(cudaFree(N_device));
    CUDA_CHECK(cudaFree(P_device));
}

void cvgpu::matrixmul(
    float* M,
    float* N,
    float* P,
    int width
)
{
    runMatrixMul(M, N, P, width, false);
}

void cvgpu::matrixmulTiled(
    float* M,
    float* N,
    float* P,
    int width
)
{
    runMatrixMul(M, N, P, width, true);
}