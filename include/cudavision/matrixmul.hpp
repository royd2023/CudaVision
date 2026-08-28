#pragma once

namespace cvgpu
{
    void matrixmul
    (
        float* M,
        float* N,
        float* P,
        int width
    );

    void matrixmulTiled
    (
        float* M,
        float* N,
        float* P,
        int width
    );
}