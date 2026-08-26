#include <iostream>
#include <chrono>

#include <cudavision/grayscale.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {
    const char* inputPath = "images/grayscale_images/input.png";
    const char* outputPath = "images/grayscale_images/output.png";

    int width;
    int height;
    int channels;

    // Load the image and force it to RGB (3 channels)
    unsigned char* input = stbi_load(
        inputPath,
        &width,
        &height,
        &channels,
        3
    );

    if (input == nullptr) {
        std::cerr << "Failed to load image: "
                  << inputPath << '\n';
        return 1;
    }

    std::cout << "Loaded image: "
              << width << " x " << height
              << '\n';

    // One grayscale value per pixel
    unsigned char* output =
        new unsigned char[width * height];
    unsigned char* tiledOutput =
        new unsigned char[width * height];

    // warmup
    cvgpu::grayscale(input, output, width, height);
    cvgpu::grayscaleTiled(input, tiledOutput, width, height);

    // timed runs
    auto start = std::chrono::high_resolution_clock::now();
    cvgpu::grayscale(
        input,
        output,
        width,
        height
    );
    auto end = std::chrono::high_resolution_clock::now();
    double grayscaleMilliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    cvgpu::grayscaleTiled(
        input,
        tiledOutput,
        width,
        height
    );
    end = std::chrono::high_resolution_clock::now();
    double tiledMilliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Unoptimized kernel: "
              << grayscaleMilliseconds << " ms\n";
    std::cout << "Tiled kernel: "
              << tiledMilliseconds << " ms\n";

    // Write grayscale PNG
    int success = stbi_write_png(
        outputPath,
        width,
        height,
        1,                  // 1 channel: grayscale
        output,
        width               // bytes per row
    );

    if (!success) {
        std::cerr << "Failed to save image\n";

        stbi_image_free(input);
        delete[] output;
        delete[] tiledOutput;
        return 1;
    }

    std::cout << "Saved grayscale image to: "
              << outputPath << '\n';

    // Clean up CPU memory
    stbi_image_free(input);
    delete[] output;
    delete[] tiledOutput;

    return 0;
}