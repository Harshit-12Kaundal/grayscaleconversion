#include <opencv2/opencv.hpp>
#include <iostream>

__global__ void rgb2gray(unsigned char* input, unsigned char* output, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int i = idx * 3;
    unsigned char r = input[i + 2]; // OpenCV loads in BGR format
    unsigned char g = input[i + 1];
    unsigned char b = input[i + 0];

    output[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
}

int main() {
    cv::Mat image = cv::imread("input.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Image not loaded.\n";
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    size_t colorBytes = width * height * 3;
    size_t grayBytes = width * height;

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, colorBytes);
    cudaMalloc(&d_output, grayBytes);

    cudaMemcpy(d_input, image.data, colorBytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (width * height + threads - 1) / threads;

    rgb2gray<<<blocks, threads>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    cv::Mat gray(height, width, CV_8UC1);
    cudaMemcpy(gray.data, d_output, grayBytes, cudaMemcpyDeviceToHost);

    cv::imwrite("output.jpg", gray);
    std::cout << "Saved grayscale image as output.jpg\n";

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
