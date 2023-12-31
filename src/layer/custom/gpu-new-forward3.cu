// Tran Tien Hoang - 20127424
#include "./gpu-new-forward.h"
#include <cmath>
#include <iostream>

#define TILE_WIDTH 16

//same as conv_forward_kernel in gpu-new-forward.cu but with shared memory optimization
__global__ void conv_forward_kernel(float *output, const float *input, const float *kernel,
                                    const int num_samples, const int output_channel, const int input_channel,
                                    const int height, const int width, const int kernel_size)
{
    const int height_out = height - kernel_size + 1;
    const int width_out = width - kernel_size + 1;
    
    int height_grid = ceil(1.0 * height_out / TILE_WIDTH);
    int width_grid = ceil(1.0 * width_out / TILE_WIDTH); 

    int batch_idx = blockIdx.x;        // batch number
    int output_feature_idx = blockIdx.y; // output feature
    int row_idx = (blockIdx.z / width_grid) * TILE_WIDTH + threadIdx.y; // row of the image matrix
    int col_idx = (blockIdx.z % width_grid) * TILE_WIDTH + threadIdx.x; // col of the image matrix

    __shared__ float shared_input[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_kernel[TILE_WIDTH][TILE_WIDTH];

    if (row_idx < height && col_idx < width)
    {
        shared_input[threadIdx.y][threadIdx.x] = input[(batch_idx * (input_channel * height * width)) + 
                                                        (threadIdx.y * (height * width)) + 
                                                        (row_idx * width) + 
                                                        col_idx];
        shared_kernel[threadIdx.y][threadIdx.x] = kernel[(output_feature_idx * (input_channel * kernel_size * kernel_size)) + 
                                                          (threadIdx.y * (kernel_size * kernel_size)) + 
                                                          (row_idx * kernel_size) + 
                                                          col_idx];
    }
    __syncthreads();

    float accumulator = 0.0f;
    if (row_idx < height_out && col_idx < width_out)
    {
        for(int i = 0; i < kernel_size; i++)
        {
            for(int j = 0; j < kernel_size; j++)
            {
                accumulator += shared_input[threadIdx.y + i][threadIdx.x + j] * shared_kernel[i][j];
            }
        }
        output[(batch_idx * (output_channel * height_out * width_out)) + 
               (output_feature_idx * (height_out * width_out)) + 
               (row_idx * width_out) + 
               col_idx] = accumulator;
    }
}


__host__ void GPUInterface::conv_forward_gpu_full(float *output_data, const float *input_data, const float *weight_data,
                                                  const int num_samples, const int output_channel, const int input_channel,
                                                  const int height_in, const int width_in, const int kernel_height)
{
    const int height_out = height_in - kernel_height + 1;
    const int width_out = width_in - kernel_height + 1;

    // Allocate device memory
    float *device_input, *device_output, *device_weight;
    cudaMalloc((void **)&device_input, num_samples * input_channel * height_in * width_in * sizeof(float));  // input features map is input_channel
    cudaMalloc((void **)&device_output, num_samples * output_channel * height_out * width_out * sizeof(float));  // output feature map is output_channel
    cudaMalloc((void **)&device_weight, output_channel * input_channel * kernel_height * kernel_height * sizeof(float));  // input_channel * output_channel filter Maps of size kernel_height * kernel_height

    // Copy input and mask data to device
    cudaMemcpy(device_input, input_data, num_samples * input_channel * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weight, weight_data, output_channel * input_channel * kernel_height * kernel_height * sizeof(float), cudaMemcpyHostToDevice);

    // Set the kernel dimensions and call the kernel
    int height_grid = ceil(1.0 * height_out / TILE_WIDTH);
    int width_grid = ceil(1.0 * width_out / TILE_WIDTH);
    int Z = height_grid * width_grid;
    dim3 num_threads_per_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 num_blocks_in_grid(num_samples, output_channel, Z);

    // Launch the kernel
    conv_forward_kernel<<<num_blocks_in_grid, num_threads_per_block>>>(device_output, device_input, device_weight, num_samples, output_channel, input_channel, height_in, width_in, kernel_height);

    // Copy the output back to host
    cudaMemcpy(output_data, device_output, num_samples * output_channel * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_weight);
}
