cmake_minimum_required(VERSION 3.24)

project(
    CudaVision
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

# ------------------------------------------------------------
# C++ / CUDA settings
# ------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Override at configure time for another GPU, for example:
# -DCMAKE_CUDA_ARCHITECTURES=86
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to compile for")

# ------------------------------------------------------------
# CudaVision library
# ------------------------------------------------------------

add_library(cudavision STATIC
    src/grayscale.cu
    src/blur.cu
    src/matrixmul.cu
)

# Public include directory.
# Anything that links against cudavision can include:
#
#   #include <cudavision/grayscale.hpp>
#
target_include_directories(cudavision
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ------------------------------------------------------------
# Example executable
# ------------------------------------------------------------

add_executable(grayscale_example
    examples/grayscale_example.cpp
)

add_executable(blur_example
    examples/blur_example.cpp
)

add_executable(matrixmul_example
    examples/matrixmul_example.cpp
)

target_include_directories(grayscale_example
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/stb
)

target_include_directories(blur_example
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/stb
)

target_link_libraries(grayscale_example
    PRIVATE
        cudavision
)

target_link_libraries(blur_example
    PRIVATE
        cudavision
)

target_link_libraries(matrixmul_example
    PRIVATE
        cudavision
)