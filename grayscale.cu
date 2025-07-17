#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
// GPU grayscale conversion kernel
__global__ void rgb2gray(unsigned char* input, unsigned char* output, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int i = idx * 3;
    unsigned char r = input[i + 2]; // OpenCV loads in BGR format
    unsigned char g = input[i + 1];
    unsigned char b = input[i + 0];

    output[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
}

// CPU grayscale conversion
void rgb2gray_cpu(const cv::Mat& input, cv::Mat& output) {
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            cv::Vec3b bgr = input.at<cv::Vec3b>(y, x);
            unsigned char gray = static_cast<unsigned char>(
                0.299f * bgr[2] + 0.587f * bgr[1] + 0.114f * bgr[0]
            );
            output.at<uchar>(y, x) = gray;
        }
    }
}

int main() {
    cv::Mat image = cv::imread("input.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Image not loaded.\n";
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    // --- CPU timing ---
    cv::Mat gray_cpu(height, width, CV_8UC1);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    rgb2gray_cpu(image, gray_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    cv::imwrite("output_cpu.jpg", gray_cpu);

    // --- GPU timing ---
    size_t colorBytes = width * height * 3;
    size_t grayBytes = width * height;
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, colorBytes);
    cudaMalloc(&d_output, grayBytes);

    cudaMemcpy(d_input, image.data, colorBytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (width * height + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    rgb2gray<<<blocks, threads>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cv::Mat gray_gpu(height, width, CV_8UC1);
    cudaMemcpy(gray_gpu.data, d_output, grayBytes, cudaMemcpyDeviceToHost);
    cv::imwrite("output_gpu.jpg", gray_gpu);

    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "CPU grayscale time: " << cpu_time << " ms\n";
    std::cout << "GPU grayscale time: " << gpu_time << " ms\n";
    std::cout << "Saved grayscale images as output_cpu.jpg and output_gpu.jpg\n";
    return 0;
}
