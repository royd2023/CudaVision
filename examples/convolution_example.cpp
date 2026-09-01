#include <iostream>
#include <chrono>
#include <cudavision/convolution.hpp>
#include <random>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    int n = argc > 1 ? std::stoi(argv[1]) : 1024;
    if (n <= 0) {
        std::cerr << "Matrix width must be positive.\n";
        return 1;
    }

    // The CUDA API expects contiguous float buffers.
    std::vector<float> M(n * n);
    std::vector<float> P(n * n);
    int r = 3;
    std::vector<float> F((2 * r + 1)*(2 * r + 1));
    // simple normalized box filter (all 2s)
    for (int i = 0; i < (2 * r + 1)*(2 * r + 1); ++i)
    {
        F[i] = 2.0f;
    }

    // Initialize the matrix and filter
    // TODO: replace with std:

    // Set up a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1, 100); // Range from 1 to 100

    // Fill the matrix with random values
    for (int i = 0; i < n * n; ++i) {
        M[i] = static_cast<float>(distrib(gen));
    }

    // warmup
    cvgpu::convolution_2D_basic(M.data(), F.data(), P.data(), r, n, n);
    cvgpu::convolution_2D_const_mem(M.data(), P.data(), r, n, n);

    // timed runs
    auto start = std::chrono::high_resolution_clock::now();
    cvgpu::convolution_2D_basic(M.data(), F.data(), P.data(), r, n, n);
    auto end = std::chrono::high_resolution_clock::now();
    double convolutionMilliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    cvgpu::convolution_2D_const_mem(M.data(), P.data(), r, n, n);
    end = std::chrono::high_resolution_clock::now();
    double convolutionConstMilliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Unoptimized kernel: "
              << convolutionMilliseconds << " ms\n";

    std::cout << "Kernel using constant memory: "
              << convolutionConstMilliseconds << " ms\n";

    return 0;
}