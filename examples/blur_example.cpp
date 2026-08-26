#include <iostream>
#include <chrono>

#include <cudavision/blur.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {
    const char* inputPath = "images/blur_images/input.png";
    const char* outputPath = "images/blur_images/output.png";

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

    // Three values per RGB pixel
    unsigned char* output =
        new unsigned char[width * height * 3];
    unsigned char* tiledOutput =
        new unsigned char[width * height * 3];

    // warmup
    cvgpu::blur(input, output, width, height);
    cvgpu::blurTiled(input, tiledOutput, width, height);

    // timed runs
    auto start = std::chrono::high_resolution_clock::now();
    cvgpu::blur(
        input,
        output,
        width,
        height
    );
    auto end = std::chrono::high_resolution_clock::now();
    double grayscaleMilliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    cvgpu::blurTiled(
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

    // Write RGB PNG
    int success = stbi_write_png(
        outputPath,
        width,
        height,
        3,                  // 3 channels: RGB
        tiledOutput,
        width * 3           // bytes per row
    );

    if (!success) {
        std::cerr << "Failed to save image\n";

        stbi_image_free(input);
        delete[] output;
        delete[] tiledOutput;
        return 1;
    }

    std::cout << "Saved blurred image to: "
              << outputPath << '\n';

    // Clean up CPU memory
    stbi_image_free(input);
    delete[] output;
    delete[] tiledOutput;

    return 0;
}