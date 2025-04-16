#include <iostream>
#include <omp.h>
#include "bicubicKernel.h"

using namespace std;

void openMP_ResizeBicubic(unsigned char* src, int srcWidth, int srcHeight, int channels,
    unsigned char* dst, int dstWidth, int dstHeight) {
    float scaleX = (float)srcWidth / dstWidth;
    float scaleY = (float)srcHeight / dstHeight;

    // compute total number of pixels
    int totalPixels = dstWidth * dstHeight;

    #pragma omp parallel
    {
        // to avoid race condition
        float localResult[4]; // 4 channels max (RGBA)

        #pragma omp for
        for (int idx = 0; idx < totalPixels; ++idx) {
            int y = idx / dstWidth; // convert linear index to 2D coordinates (y)
            int x = idx % dstWidth; // convert linear index to 2D coordinates (x)

            for (int c = 0; c < channels; ++c) {
                float srcX = x * scaleX;
                float srcY = y * scaleY;
                int x1 = (int)srcX;
                int y1 = (int)srcY;

                localResult[c] = 0.0f; // reset local result

                // sum result using thread-local variable
                for (int m = -1; m <= 2; ++m) {
                    for (int n = -1; n <= 2; ++n) {
                        float weight = bicubicKernel(srcX - (x1 + n)) * bicubicKernel(srcY - (y1 + m));
                        float pixelValue = getPixelValue(src, srcWidth, srcHeight, channels, x1 + n, y1 + m, c);
                        localResult[c] += pixelValue * weight;
                    }
                }

                // write the computed result to the output image
                dst[(y * dstWidth + x) * channels + c] = min(max((int)localResult[c], 0), 255);
            }
        }
    }
}

