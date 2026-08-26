# CudaVision

CudaVision is a small CUDA image-processing library with grayscale and blur
kernels, including tiled implementations for comparison.

## Status

This project is experimental and currently version `0.1.0`. The public API and
build system may change as the library develops.

## Requirements

- NVIDIA GPU with a compatible CUDA toolkit and driver
- CMake 3.24 or newer
- C++17-compatible compiler
- Visual Studio on Windows, or a CUDA-supported C++ toolchain on another platform

## Build

Configure and build from the repository root:

```powershell
cmake -S . -B build
cmake --build build --config Debug
```

The examples are built as `grayscale_example` and `blur_example`. On a
multi-configuration generator such as Visual Studio, run them with:

```powershell
.\build\Debug\grayscale_example.exe
.\build\Debug\blur_example.exe
```

The examples read input images from `images/` and write generated output images
there. Build files and generated outputs are excluded by `.gitignore`.

## Library API

Include the public headers and link the `cudavision` target:

```cpp
#include <cudavision/grayscale.hpp>

cvgpu::grayscale(input, output, width, height);
cvgpu::grayscaleTiled(input, output, width, height);
```

The input is an interleaved RGB image buffer. The output is one grayscale byte
per pixel. `blur.hpp` provides the corresponding `blur` and `blurTiled`
functions. Callers own the input and output buffers, and must keep them valid
for the duration of each call.

## Third-party code

The examples use [stb_image](https://github.com/nothings/stb) and
`stb_image_write`, which are public-domain libraries included under
`external/stb/`.

## License

No project license has been selected yet. Add a `LICENSE` file before treating
this repository as a reusable public library.
