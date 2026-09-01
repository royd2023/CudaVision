#include <iostream>
#include <chrono>
#include <cudavision/matrixmul.hpp>
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
    std::vector<float> N(n * n);
    std::vector<float> P(n * n);

    // Set up a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1, 100); // Range from 1 to 100

    // Fill the matrix with random values
    for (int i = 0; i < n * n; ++i) {
        M[i] = static_cast<float>(distrib(gen));
        N[i] = static_cast<float>(distrib(gen));
    }

    // warmup
    cvgpu::matrixmul(M.data(), N.data(), P.data(), n);
    cvgpu::matrixmulTiled(M.data(), N.data(), P.data(), n);

    // timed runs
    auto start = std::chrono::high_resolution_clock::now();
    cvgpu::matrixmul(M.data(), N.data(), P.data(), n);
    auto end = std::chrono::high_resolution_clock::now();
    double matrixmulMilliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    cvgpu::matrixmulTiled(M.data(), N.data(), P.data(), n);
    end = std::chrono::high_resolution_clock::now();
    double matrixmulTiledMilliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Unoptimized kernel: "
              << matrixmulMilliseconds << " ms\n";
    std::cout << "Tiled kernel: "
              << matrixmulTiledMilliseconds << " ms\n";
    

    return 0;
}