#include <iostream>
#include <cmath>

using namespace std;

float bicubicKernel(float d) {
    d = fabs(d);
    if (d <= 1.0f) {
        return (1.5f * d * d * d - 2.5f * d * d + 1.0f);
    }
    else if (d <= 2.0f) {
        return (-0.5f * d * d * d + 2.5f * d * d - 4.0f * d + 2.0f);
    }
    return 0.0f;
}

float getPixelValue(unsigned char* image, int width, int height, int channels, int x, int y, int c) {
    x = max(0, min(x, width - 1));
    y = max(0, min(y, height - 1));
    return image[(y * width + x) * channels + c];
}