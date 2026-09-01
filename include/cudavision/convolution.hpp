#pragma once

namespace cvgpu
{
    void convolution_2D_basic
    (
        float *input,
        float *filter,
        float *output,
        int radius,
        int width,
        int height
    );

    void convolution_2D_const_mem
    (
        float *input,
        float *output,
        int radius,
        int width,
        int height
    );

    void convolution_tiled_2D_const_mem
    (
        float *input,
        float *output,
        int width,
        int height
    );

    void convolution_cached_tiled_2D_const_mem
    (
        float *input,
        float *output,
        int width,
        int height
    );
}