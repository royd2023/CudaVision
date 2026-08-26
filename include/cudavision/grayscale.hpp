#pragma once

namespace cvgpu 
{
    void grayscale
    (
        const unsigned char* input,
        unsigned char* output,
        int width,
        int height
    );

    void grayscaleTiled
    (
        const unsigned char* input,
        unsigned char* output,
        int width,
        int height
    );
}