#include <iostream>
#include "bicubicKernel.h"

using namespace std;

void serial_ResizeBicubic(unsigned char* src, int srcWidth, int srcHeight, int channels,
    unsigned char* dst, int dstWidth, int dstHeight) {
    float scaleX = (float)srcWidth / dstWidth;
    float scaleY = (float)srcHeight / dstHeight;

    for (int y = 0; y < dstHeight; ++y) {
        for (int x = 0; x < dstWidth; ++x) {
            for (int c = 0; c < channels; ++c) {
                float srcX = x * scaleX;
                float srcY = y * scaleY;
                int x1 = (int)srcX;
                int y1 = (int)srcY;

                float result = 0.0f;
                for (int m = -1; m <= 2; ++m) {
                    for (int n = -1; n <= 2; ++n) {
                        float weight = bicubicKernel(srcX - (x1 + n)) * bicubicKernel(srcY - (y1 + m));
                        result += getPixelValue(src, srcWidth, srcHeight, channels, x1 + n, y1 + m, c) * weight;
                    }
                }

                dst[(y * dstWidth + x) * channels + c] = min(max((int)result, 0), 255);
            }
        }
    }
}


