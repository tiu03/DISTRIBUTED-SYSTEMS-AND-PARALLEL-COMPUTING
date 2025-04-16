#include <omp.h>
#include <string>
#include <boost/tuple/tuple.hpp>
#include "gnuplot-iostream.h"

#include "stb_image.h"
#include "stb_image_write.h"

#include "simpleResize.h"
#include "serial_ResizeBicubic.h"
#include "openMP_ResizeBicubic.h"
#include "openCL_ResizeBicubic.h"
#include "cuda_ResizeBicubic.cuh"

using namespace std;

// Function to load the image
unsigned char* loadImage(const char* filename, int& width, int& height, int& channels) {
    unsigned char* img = stbi_load(filename, &width, &height, &channels, 0);
    if (!img) {
        cerr << "Failed to load image: " << filename << endl;
    }
    return img;
}

// Function to save the image
bool saveImage(const char* filename, unsigned char* imgData, int width, int height, int channels) {
    if (!stbi_write_png(filename, width, height, channels, imgData, width * channels)) {
        cerr << "Failed to save image: " << filename << endl;
        return false;
    }
    return true;
}

// Function to resize image using a specific method
double resizeImage(void (*resizeFunc)(unsigned char*, int, int, int, unsigned char*, int, int),
    const char* inputFileName, const char* outputFileName, int newWidth, int newHeight) {
    int width, height, channels;
    unsigned char* img = loadImage(inputFileName, width, height, channels);
    if (!img) {
        return -1;
    }

    unsigned char* resizedImg = new unsigned char[newWidth * newHeight * channels];
    double start_time = omp_get_wtime();
    resizeFunc(img, width, height, channels, resizedImg, newWidth, newHeight);
    double run_time = omp_get_wtime() - start_time;

    if (!saveImage(outputFileName, resizedImg, newWidth, newHeight, channels)) {
        stbi_image_free(img);
        delete[] resizedImg;
        return -1;
    }

    stbi_image_free(img);
    delete[] resizedImg;
    return run_time;
}

// Function to calculate Mean Squared Error (MSE) between two images
double calculateMSE(const unsigned char* img1, const unsigned char* img2, int width, int height, int channels) {
    double mse = 0.0;
    int totalPixels = width * height * channels;

#pragma omp parallel for
    for (int i = 0; i < totalPixels; ++i) {
        double diff = (double)img1[i] - (double)img2[i];
        mse += diff * diff;
    }

    return mse / totalPixels;
}

// Function to generate output file names based on input file and method
string generateOutputFileName(const string& inputFileName, const string& method, int width) {
    size_t lastDot = inputFileName.find_last_of(".");
    string baseName = inputFileName.substr(0, lastDot);
    string format = inputFileName.substr(lastDot, inputFileName.length());
    return baseName + "_" + method + "_" + to_string(width) + format;
}

// Function to resize using all methods and widths
void experiment_processImage(const char* inputFileName) {
    //int widths[] = { 3000, 6000, 12000, 24000 };
    int widths[] = { 1000, 2000, 3000 };
    string methods[] = { "serial", "openmp", "opencl", "cuda" };

    // Load the original image once to get the aspect ratio
    int originalWidth, originalHeight, originalChannels;
    unsigned char* imgOriginal = loadImage(inputFileName, originalWidth, originalHeight, originalChannels);

    if (!imgOriginal) {
        cerr << "Error: Could not load the image to determine aspect ratio." << endl;
        return;
    }

    // Calculate the aspect ratio (width-to-height ratio)
    double aspectRatio = static_cast<double>(originalHeight) / static_cast<double>(originalWidth);

    // Now process the image for each width
    for (int width : widths) {
        string output;
        // Calculate the new height while preserving the aspect ratio
        int newHeight = static_cast<int>(width * aspectRatio);

        // Process using Serial method
        output = generateOutputFileName(inputFileName, "serial", width);
        string outputSerial = "experiment output/" + output.substr(5, output.length());
        double serialTime = resizeImage(serial_ResizeBicubic, inputFileName, outputSerial.c_str(), width, newHeight);
        cout << endl << "Serial resize for width " << width << " (height " << newHeight << ") took " << serialTime << " seconds." << endl;

        // Process using OpenMP method
        output = generateOutputFileName(inputFileName, "openmp", width);
        string outputOpenMP = "output/" + output.substr(5, output.length());
        double openmpTime = resizeImage(openMP_ResizeBicubic, inputFileName, outputOpenMP.c_str(), width, newHeight);
        cout << "OpenMP resize for width " << width << " (height " << newHeight << ") took " << openmpTime << " seconds." << endl;

        // Process using OpenCL method
        output = generateOutputFileName(inputFileName, "opencl", width);
        string outputOpenCL = "output/" + output.substr(5, output.length());
        double openclTime = resizeImage(openCL_ResizeBicubic, inputFileName, outputOpenCL.c_str(), width, newHeight);
        cout << "OpenCL resize for width " << width << " (height " << newHeight << ") took " << openclTime << " seconds." << endl;

        // Process using CUDA method
        output = generateOutputFileName(inputFileName, "cuda", width);
        string outputCUDA = "output/" + output.substr(5, output.length());
        double cudaTime = resizeImage(cuda_ResizeBicubic, inputFileName, outputCUDA.c_str(), width, newHeight);
        cout << "CUDA   resize for width " << width << " (height " << newHeight << ") took " << cudaTime << " seconds." << endl;

        // Load and compare the resized images for MSE (as before)
        unsigned char* imgSerial = loadImage(outputSerial.c_str(), originalWidth, originalHeight, originalChannels);
        unsigned char* imgOpenMP = loadImage(outputOpenMP.c_str(), originalWidth, originalHeight, originalChannels);
        unsigned char* imgOpenCL = loadImage(outputOpenCL.c_str(), originalWidth, originalHeight, originalChannels);
        unsigned char* imgCUDA = loadImage(outputCUDA.c_str(), originalWidth, originalHeight, originalChannels);

        output = generateOutputFileName(inputFileName, "simple", width);
        string outputSimple = "output/" + output.substr(5, output.length());
        resizeImage(simple_Resize, inputFileName, outputSimple.c_str(), width, newHeight);

        double mseOpenMP = calculateMSE(imgSerial, imgOpenMP, originalWidth, originalHeight, originalChannels);
        double mseOpenCL = calculateMSE(imgSerial, imgOpenCL, originalWidth, originalHeight, originalChannels);
        double mseCUDA = calculateMSE(imgSerial, imgCUDA, originalWidth, originalHeight, originalChannels);

        cout << endl << "MSE between serial and OpenMP for width " << width << ": " << mseOpenMP << endl;
        cout << "MSE between serial and OpenCL for width " << width << ": " << mseOpenCL << endl;
        cout << "MSE between serial and CUDA   for width " << width << ": " << mseCUDA << endl;

        cout << endl << "------------------------------------------------------------------------" << endl;

        // Free image memory
        stbi_image_free(imgSerial);
        stbi_image_free(imgOpenMP);
        stbi_image_free(imgOpenCL);
        stbi_image_free(imgCUDA);
    }

    // Free the original image memory
    stbi_image_free(imgOriginal);
}

// Function to resize using all methods and widths
void specific_processImage(const char* inputFileName, int width, int int_method) {
    // Load the original image once to get the aspect ratio
    int originalWidth, originalHeight, originalChannels;
    unsigned char* imgOriginal = loadImage(inputFileName, originalWidth, originalHeight, originalChannels);
    string method;
    if (!imgOriginal) {
        cerr << "Error: Could not load the image to determine aspect ratio." << endl;
        return;
    }

    // Calculate the aspect ratio (width-to-height ratio)
    double aspectRatio = static_cast<double>(originalHeight) / static_cast<double>(originalWidth);
    string output;
    int newHeight = static_cast<int>(width * aspectRatio);

    switch (int_method) {
    case 1: {
        method = "openmp";
        output = generateOutputFileName(inputFileName, method, width);
        string outputOpenmp = "output/" + output.substr(5, output.length());
        double time = resizeImage(openMP_ResizeBicubic, inputFileName, outputOpenmp.c_str(), width, newHeight);
        cout << endl << "Resize for width " << width << " (height " << newHeight << ") took " << time << " seconds." << endl;
        break;
    }
    case 2: {
        method = "cuda";
        output = generateOutputFileName(inputFileName, method, width);
        string outputCUDA = "output/" + output.substr(5, output.length());
        double time = resizeImage(cuda_ResizeBicubic, inputFileName, outputCUDA.c_str(), width, newHeight);
        cout << endl << "Resize for width " << width << " (height " << newHeight << ") took " << time << " seconds." << endl;
        break;
    }
    default: {
        return;
    }
    }
    cout << endl << "------------------------------------------------------------------------" << endl;

    // Free the original image memory
    stbi_image_free(imgOriginal);
}