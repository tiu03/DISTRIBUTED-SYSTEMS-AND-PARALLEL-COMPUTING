void simple_Resize(unsigned char* src, int srcWidth, int srcHeight, int channels,
    unsigned char* dst, int dstWidth, int dstHeight) {
    float scaleX = (float)srcWidth / dstWidth;
    float scaleY = (float)srcHeight / dstHeight;

    #pragma omp parallel for
    for (int y = 0; y < dstHeight; ++y) {
        for (int x = 0; x < dstWidth; ++x) {
            int srcX = (int)(x * scaleX);
            int srcY = (int)(y * scaleY);

            for (int c = 0; c < channels; ++c) {
                dst[(y * dstWidth + x) * channels + c] = src[(srcY * srcWidth + srcX) * channels + c];
            }
        }
    }
}