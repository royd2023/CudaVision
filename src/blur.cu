#include "cudavision/blur.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CHANNELS 3
#define BLUR_SIZE 10
#define TILE_WIDTH 16
#define IN_TILE_WIDTH (TILE_WIDTH + 2*BLUR_SIZE)

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

__global__ void blurKernel
(
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
        for (int c = 0; c < CHANNELS; c++)
        {
            int pixVal = 0;
            int pixels = 0;
            // get average of the surrounding BLUR_SIZE x BLUR_SIZE box
            for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow)
            {
                for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol)
                {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;

                    // Verify we have a valid image pixel
                    if (curRow>=0 && curRow < height && curCol >= 0 && curCol < width) 
                    {
                        pixVal += input[(curRow*width + curCol)*CHANNELS + c];
                        pixels++; // Keep track of number of pixels in average
                    }
                }
            }
            // Write our new pixel value out
            output[(row*width + col)*CHANNELS + c] = (unsigned char)(pixVal/pixels);
        }
    }
}

__global__ void tiledBlurKernel
(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
)
{
    __shared__ unsigned char tile[IN_TILE_WIDTH][IN_TILE_WIDTH][CHANNELS];

    int tileElements = IN_TILE_WIDTH * IN_TILE_WIDTH;
    int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;
    // tile[0][0] = (tileOriginCol, tileOriginRow)
    int tileOriginCol = blockIdx.x * TILE_WIDTH - BLUR_SIZE;
    int tileOriginRow = blockIdx.y * TILE_WIDTH - BLUR_SIZE;

    // loops through all the threads in the IN_TILE_WIDTH * IN_TILE_WIDTH shared memory tile,
    // figure out where it lives in the tile, where it lives in the input image, then copy it over to shared memory
    for (int tileIndex = threadIndex; tileIndex < tileElements;
         tileIndex += blockDim.x * blockDim.y)
    {
        int tileRow = tileIndex / IN_TILE_WIDTH;
        int tileCol = tileIndex % IN_TILE_WIDTH;
        int globalRow = tileOriginRow + tileRow;
        int globalCol = tileOriginCol + tileCol;
        bool inBounds = globalRow >= 0 && globalRow < height &&
                        globalCol >= 0 && globalCol < width;

        for (int c = 0; c < CHANNELS; c++)
        {
            // copies pixel values from input image into shared memory
            tile[tileRow][tileCol][c] = inBounds
                ? input[(globalRow * width + globalCol) * CHANNELS + c]
                : 0;
        }
    }

    __syncthreads();

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    bool inBounds = row < height && col < width;

    if (!inBounds) {
        return;
    }

    for (int c = 0; c < CHANNELS; c++)
    {
        int pixVal = 0;
        int pixels = 0;
        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow)
        {
            for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol)
            {
                int globalRow = row + blurRow;
                int globalCol = col + blurCol;
                int tileRow = threadIdx.y + BLUR_SIZE + blurRow;
                int tileCol = threadIdx.x + BLUR_SIZE + blurCol;

                if (globalRow >= 0 && globalRow < height &&
                    globalCol >= 0 && globalCol < width)
                {
                    pixVal += tile[tileRow][tileCol][c];
                    pixels++;
                }
            }
        }

        output[(row * width + col) * CHANNELS + c] =
            (unsigned char)(pixVal / pixels);
    }
}

void runBlur(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    bool tiled
) 
{
    size_t size = width * height * CHANNELS * sizeof(unsigned char);
    unsigned char *input_device, *output_device;
    
    // GPU allocation
    CUDA_CHECK(cudaMalloc((void**) &input_device, size));
    CUDA_CHECK(cudaMalloc((void**) &output_device, size));

    // cudaMemcpy
    CUDA_CHECK(cudaMemcpy(input_device, input, size, cudaMemcpyHostToDevice));

    // kernel launch
    constexpr int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
        (height + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    if (tiled) {
        tiledBlurKernel<<<gridDim, blockDim>>>(
            input_device, output_device, width, height
        );
    } else {
        blurKernel<<<gridDim, blockDim>>>(
            input_device, output_device, width, height
        );
    }

    // Check whether the kernel launch itself was valid
    CUDA_CHECK(cudaGetLastError());

    // Wait for the kernel and catch execution errors
    CUDA_CHECK(cudaDeviceSynchronize());

    // cudaMemcpy back
    CUDA_CHECK(cudaMemcpy(output, output_device, size, cudaMemcpyDeviceToHost));

    // free memory
    CUDA_CHECK(cudaFree(input_device));
    CUDA_CHECK(cudaFree(output_device));
}

void cvgpu::blur(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
)
{
    runBlur(input, output, width, height, false);
}

void cvgpu::blurTiled(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
)
{
    runBlur(input, output, width, height, true);
}