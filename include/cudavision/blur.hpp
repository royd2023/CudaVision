#pragma once

namespace cvgpu 
{
    void blur
    (
        const unsigned char* input,
        unsigned char* output,
        int width,
        int height
    );

    void blurTiled
    (
        const unsigned char* input,
        unsigned char* output,
        int width,
        int height
    );
}