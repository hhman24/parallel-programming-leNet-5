#include "./gpu-new-forward.h"
#include <cmath>
#include <iostream>

#define TILE_WIDTH_CONST 16
#define TILE_WIDTH_SHARED_C1 16
#define TILE_WIDTH_SHARED_C3 12

__constant__ float deviceMaskData[2400];

__global__ void conv_forward_kernel(float* output, const float* input, const int num_samples,
                                    const int output_channel, const int input_channel,
                                    const int height, const int width, const int kernel_size)
{
    int TILE_WIDTH_SHARED;
    if (input_channel == 1){
        TILE_WIDTH_SHARED = TILE_WIDTH_SHARED_C1;
    }
    else{
        TILE_WIDTH_SHARED = TILE_WIDTH_SHARED_C3;
    }

    extern __shared__ float shared_input[];

    const int H_out = height - kernel_size + 1;
    const int W_out = width - kernel_size + 1;

    int W_grid = ceil(1.0 * W_out / TILE_WIDTH_SHARED);

    int b = blockIdx.x;                 // batch number
    int m = blockIdx.y;                 // output feature
    int ty = threadIdx.y;               // thread ID in the current TILE  
    int tx = threadIdx.x;
    
    int h = (blockIdx.z / W_grid) * TILE_WIDTH_SHARED + ty; // row of the input image matrix
    int w = (blockIdx.z % W_grid) * TILE_WIDTH_SHARED + tx; // col of the input image matrix

    int startOfTile_h = (blockIdx.z / W_grid) * TILE_WIDTH_SHARED; // row of the input image matrix
    int startOfTile_w = (blockIdx.z % W_grid) * TILE_WIDTH_SHARED; // col of the input image matrix
    for (int c = 0; c < input_channel; c++)
    {
        for(int i = ty; i < TILE_WIDTH_SHARED + kernel_size - 1; i += TILE_WIDTH_SHARED)
        {
            for(int j = tx; j < TILE_WIDTH_SHARED + kernel_size - 1; j += TILE_WIDTH_SHARED)
            {
                if (startOfTile_h + i < height && startOfTile_w + j < width)
                {
                    shared_input[c * (TILE_WIDTH_SHARED + kernel_size - 1) * (TILE_WIDTH_SHARED + kernel_size - 1) + i * (TILE_WIDTH_SHARED + kernel_size - 1) + j] = input[b * (input_channel * height * width) + c * (height * width) + (startOfTile_h + i) * width + startOfTile_w + j];
                }
            }
        }
    }
    __syncthreads();

    if ((h < H_out) && (w < W_out)) 
    {
        float accum = 0.0f;
        for(int c=0; c<input_channel; c++)             // sum over all input features
        {
            for(int p=0; p< kernel_size; p++)         // KxK filter 
                for(int q=0; q< kernel_size; q++)
                    accum += shared_input[c * (TILE_WIDTH_SHARED + kernel_size - 1) * (TILE_WIDTH_SHARED + kernel_size - 1) + (p+ty) * (TILE_WIDTH_SHARED + kernel_size - 1) + (q+tx)] * deviceMaskData[m * (input_channel * kernel_size * kernel_size) + c * (kernel_size * kernel_size) + p * kernel_size + q]; 
        }
        output[b * (output_channel * H_out * W_out) + m * (H_out * W_out) + h * W_out + w] = accum;
    }
}

__host__ void GPUInterface::conv_forward_gpu_full(float *output_data, const float *input_data, const float *weight_data,
                                                  const int num_samples, const int output_channel, const int input_channel,
                                                  const int height_in, const int width_in, const int kernel_height)
{
    int TILE_WIDTH_SHARED;
    if (input_channel == 1){
        TILE_WIDTH_SHARED = TILE_WIDTH_SHARED_C1;
    }
    else{
        TILE_WIDTH_SHARED = TILE_WIDTH_SHARED_C3;
    }
    std::cout << ". Combined Optimization:\n";

    const int H_out = height_in - kernel_height + 1;
    const int W_out = width_in - kernel_height + 1;

    int inputSize = num_samples * input_channel * height_in * width_in * sizeof(float);
    int outputSize = num_samples * output_channel * H_out * W_out * sizeof(float);
    int maskSize = output_channel * input_channel * kernel_height * kernel_height * sizeof(float);

    float *device_input, *device_output;

    cudaMalloc((void **)&device_input, inputSize);
    cudaMalloc((void **)&device_output, outputSize);

    cudaMemcpy(device_input, input_data, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceMaskData, weight_data, maskSize);

    dim3 numThreadsPerBlock, numBlocksInGrid;

    numThreadsPerBlock = dim3(TILE_WIDTH_SHARED, TILE_WIDTH_SHARED, 1);
    int shmem_size = input_channel * (TILE_WIDTH_SHARED + kernel_height - 1) * (TILE_WIDTH_SHARED + kernel_height - 1) * sizeof(float);
    numBlocksInGrid = dim3(num_samples, output_channel, ceil(1.0 * H_out / TILE_WIDTH_SHARED)*ceil(1.0 * W_out / TILE_WIDTH_SHARED));
    
    conv_forward_kernel<<<numBlocksInGrid, numThreadsPerBlock, shmem_size>>>(device_output, device_input, num_samples, output_channel, input_channel, height_in, width_in, kernel_height);

    cudaMemcpy(output_data, device_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);
}
