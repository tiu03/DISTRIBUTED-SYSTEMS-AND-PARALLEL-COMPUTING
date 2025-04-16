#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include <omp.h>
#include <string>
#include <boost/tuple/tuple.hpp>
#include <numeric>
#include "gnuplot-iostream.h"

#include "stb_image.h"
#include "stb_image_write.h"

#include "simpleResize.h"
#include "serial_ResizeBicubic.h"
#include "openMP_ResizeBicubic.h"
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

// Function to process image with different methods and collect performance data
void experiment_processImage(const char* inputFileName, int w[]) {
    cout << endl << "------------------------------------------------------------------------" << endl;

    int widths[] = {w[0], w[1], w[2], w[3], w[4]};
    int ctr = 0;
    vector<string> methods = { "serial", "openmp", "cuda" };
    vector<double> serial_exec_time = { 0, 0, 0, 0, 0 };
    vector<double> openmp_exec_time = { 0, 0, 0, 0, 0 };
    vector<double> openmp_pg_result = { 0, 0, 0, 0, 0 };
    vector<double> cuda_exec_time = { 0, 0, 0, 0, 0 };
    vector<double> cuda_pg_result = { 0, 0, 0, 0, 0 };

    const int numTrials = 5;

    // Load the original image once to get the aspect ratio
    int originalWidth, originalHeight, originalChannels;
    unsigned char* imgOriginal = loadImage(inputFileName, originalWidth, originalHeight, originalChannels);

    if (!imgOriginal) {
        cerr << "Error: Could not load the image to determine aspect ratio." << endl;
        return;
    }

    // Calculate the aspect ratio (width-to-height ratio)
    double aspectRatio = static_cast<double>(originalHeight) / static_cast<double>(originalWidth);

    for (int width : widths) {
        if (width == 0) {
            cout << "Message: no more images...";
            break;
        }
        int newHeight = static_cast<int>(width * aspectRatio);

        vector<double> timesSerial(numTrials);
        vector<double> timesOpenMP(numTrials);
        vector<double> timesCUDA(numTrials);
        vector<double> mseOpenMP(numTrials);
        vector<double> mseCUDA(numTrials);

        bool validResults = true;

        // Perform trials
        for (int trial = 0; trial < numTrials; ++trial) {
            // Process using Serial method
            string output = generateOutputFileName(inputFileName, "serial", width);
            string outputSerial = "output/" + output.substr(5, output.length());
            double serialTime = resizeImage(serial_ResizeBicubic, inputFileName, outputSerial.c_str(), width, newHeight);
            timesSerial[trial] = serialTime;

            // Process using OpenMP method
            output = generateOutputFileName(inputFileName, "openmp", width);
            string outputOpenMP = "output/" + output.substr(5, output.length());
            double openmpTime = resizeImage(openMP_ResizeBicubic, inputFileName, outputOpenMP.c_str(), width, newHeight);
            timesOpenMP[trial] = openmpTime;

            // Process using CUDA method
            output = generateOutputFileName(inputFileName, "cuda", width);
            string outputCUDA = "output/" + output.substr(5, output.length());
            double cudaTime = resizeImage(cuda_ResizeBicubic, inputFileName, outputCUDA.c_str(), width, newHeight);
            timesCUDA[trial] = cudaTime;

            // Process using Simple method (not bicubic)
            output = generateOutputFileName(inputFileName, "simple", width);
            string outputSimple = "output/" + output.substr(5, output.length());
            double simpleTime = resizeImage(simple_Resize, inputFileName, outputSimple.c_str(), width, newHeight);

            // Load and compare the resized images for MSE
            unsigned char* imgSerial = loadImage(outputSerial.c_str(), originalWidth, originalHeight, originalChannels);
            unsigned char* imgOpenMP = loadImage(outputOpenMP.c_str(), originalWidth, originalHeight, originalChannels);
            unsigned char* imgCUDA = loadImage(outputCUDA.c_str(), originalWidth, originalHeight, originalChannels);

            mseOpenMP[trial] = calculateMSE(imgSerial, imgOpenMP, originalWidth, originalHeight, originalChannels);
            mseCUDA[trial] = calculateMSE(imgSerial, imgCUDA, originalWidth, originalHeight, originalChannels);

            // Check if the results are valid
            if (mseOpenMP[trial] > 0 || mseCUDA[trial] > 0) {
                cout << "Invalid results detected. Stopping further trials." << endl;
                validResults = false;
                break;
            }

            stbi_image_free(imgSerial);
            stbi_image_free(imgOpenMP);
            stbi_image_free(imgCUDA);
        }

        if (validResults) {
            // Calculate averages and performance gains
            double avgSerialTime = accumulate(timesSerial.begin(), timesSerial.end(), 0.0) / numTrials;
            double avgOpenMPTime = accumulate(timesOpenMP.begin(), timesOpenMP.end(), 0.0) / numTrials;
            double avgCUDA = accumulate(timesCUDA.begin(), timesCUDA.end(), 0.0) / numTrials;

            double avgMSEOpenMP = accumulate(mseOpenMP.begin(), mseOpenMP.end(), 0.0) / numTrials;
            double avgMSECuda = accumulate(mseCUDA.begin(), mseCUDA.end(), 0.0) / numTrials;

            double performanceGainOpenMP = avgSerialTime / avgOpenMPTime;
            double performanceGainCUDA = avgSerialTime / avgCUDA;

            cout << fixed << setprecision(4);
            cout << endl << "Width: " << width << endl;
            cout << "Serial average time: " << avgSerialTime << " seconds." << endl;
            cout << "OpenMP average time: " << avgOpenMPTime << " seconds. Performance gain: " << performanceGainOpenMP << endl;
            cout << "CUDA average time: " << avgCUDA << " seconds. Performance gain: " << performanceGainCUDA << endl;
            cout << endl << "------------------------------------------------------------------------" << endl;
            serial_exec_time[ctr] = avgSerialTime;
            openmp_exec_time[ctr] = avgOpenMPTime;
            openmp_pg_result[ctr] = performanceGainOpenMP;
            cuda_exec_time[ctr] = avgCUDA;
            cuda_pg_result[ctr] = performanceGainCUDA;
        }
        ctr = ctr + 1;
    }

    Gnuplot gp;

    // Save Histogram for average performance gain as PNG
    gp << "set terminal png size 800,600\n"; // Set terminal to PNG
    gp << "set output 'plot/performance_gain_comparison.png'\n";
    gp << "set title 'Performance Gain Comparison (OpenMP vs CUDA)'\n";
    gp << "set ylabel 'Performance Gain'\n";
    gp << "set xlabel 'Image Width'\n";
    gp << "set style data histogram\n";
    gp << "set style histogram clustered gap 1\n";
    gp << "set style fill solid border -1\n";
    gp << "set boxwidth 0.9 relative\n";
    gp << "plot '-' using 2:xtic(1) title 'OpenMP', '-' using 2:xtic(1) title 'CUDA'\n";
    gp.send1d(boost::make_tuple(widths, openmp_pg_result));
    gp.send1d(boost::make_tuple(widths, cuda_pg_result));

    // Save line plot for execution time comparison as PNG
    gp << "set output 'plot/execution_time_comparison.png'\n";
    gp << "set title 'Execution Time Comparison (OpenMP vs CUDA vs Serial)'\n";
    gp << "set ylabel 'Execution Time (s)'\n";
    gp << "set xlabel 'Image Width'\n";
    gp << "set style data linespoints\n";
    gp << "plot '-' using 1:2 with linespoints title 'Serial', '-' using 1:2 with linespoints title 'OpenMP', '-' using 1:2 with linespoints title 'CUDA'\n";
    gp.send1d(boost::make_tuple(widths, serial_exec_time));
    gp.send1d(boost::make_tuple(widths, openmp_exec_time));
    gp.send1d(boost::make_tuple(widths, cuda_exec_time));

    // Save line plot for performance gain comparison as PNG
    gp << "set output 'plot/performance_gain_lineplot.png'\n"; \
    gp << "set title 'Performance Gain Comparison (OpenMP vs CUDA)'\n";
    gp << "set ylabel 'Performance Gain'\n";
    gp << "set xlabel 'Image Width'\n";
    gp << "set style data linespoints\n";
    gp << "plot '-' using 1:2 with linespoints title 'OpenMP PG', '-' using 1:2 with linespoints title 'CUDA PG'\n";
    gp.send1d(boost::make_tuple(widths, openmp_pg_result));
    gp.send1d(boost::make_tuple(widths, cuda_pg_result));

    // Close the output file
    gp << "set output\n";  // Reset output

    stbi_image_free(imgOriginal);
}

int main() {
    string inputFileName;
    int mode;

    cout << "       Image Processing Application" << endl;
    cout << "------------------------------------------" << endl;

    cout << "Enter the input image name: ";
    getline(cin, inputFileName);
    inputFileName = "data/" + inputFileName;

    int widths[] = {0,0,0,0,0};
    string input;
    int count = 0;

    cout << "Enter width(s) values separated by commas in integer (exp: 1,2,3.. [max 5]): ";
    getline(cin, input);

    std::stringstream ss(input);
    int value;
    char comma;

    // Extract integers and ignore commas
    while (ss >> value && count < 5) {
        widths[count++] = value;
        ss >> comma;
    }

    experiment_processImage(inputFileName.c_str(), widths);

    return 0;
}