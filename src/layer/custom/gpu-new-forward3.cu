// Tran Tien Hoang - 20127424
#include "./gpu-new-forward.h"
#include <cmath>
#include <iostream>

#define TILE_WIDTH 16
#define KERNEL_SIZE_MAX 5 // Assume maximum kernel size is 5x5

__global__ void conv_forward_kernel(float *output, const float *input, const float *kernel,
                                    const int num_samples, const int output_channel, const int input_channel,
                                    const int height, const int width, const int kernel_size)
{
    const int height_out = height - kernel_size + 1;
    const int width_out = width - kernel_size + 1;

    int height_grid = ceil(1.0 * height_out / TILE_WIDTH);
    int width_grid = ceil(1.0 * width_out / TILE_WIDTH); 

    // Declare shared memory for input tile and kernel tile
    __shared__ float input_tile[TILE_WIDTH + KERNEL_SIZE_MAX - 1][TILE_WIDTH + KERNEL_SIZE_MAX - 1];
    __shared__ float kernel_tile[KERNEL_SIZE_MAX][KERNEL_SIZE_MAX];

    // Calculate block and thread indices
    int batch_idx = blockIdx.x;
    int output_feature_idx = blockIdx.y;
    int block_z = blockIdx.z;
    int block_row = block_z / width_grid;
    int block_col = block_z % width_grid;
    int row_idx = block_row * TILE_WIDTH + threadIdx.y;
    int col_idx = block_col * TILE_WIDTH + threadIdx.x;

    float accumulator = 0.0f;

    // Iterate over input channels
    for (int input_channel_idx = 0; input_channel_idx < input_channel; input_channel_idx++)
    {
        // Copy input tile to shared memory
        if (row_idx < height && col_idx < width)
        {
            input_tile[threadIdx.y][threadIdx.x] = input[(batch_idx * (input_channel * height * width)) +
                                                         (input_channel_idx * (height * width)) +
                                                         (row_idx * width) +
                                                         col_idx];
        }
        else
        {
            input_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Copy kernel tile to shared memory
        if (threadIdx.y < kernel_size && threadIdx.x < kernel_size)
        {
            kernel_tile[threadIdx.y][threadIdx.x] = kernel[(output_feature_idx * (input_channel * kernel_size * kernel_size)) +
                                                           (input_channel_idx * (kernel_size * kernel_size)) +
                                                           (threadIdx.y * kernel_size) +
                                                           threadIdx.x];
        }

        __syncthreads();

        // Convolution computation using shared memory
        for (int k_row = 0; k_row < kernel_size; k_row++)
        {
            for (int k_col = 0; k_col < kernel_size; k_col++)
            {
                accumulator += input_tile[threadIdx.y + k_row][threadIdx.x + k_col] * kernel_tile[k_row][k_col];
            }
        }

        __syncthreads();
    }

    // Store result in output
    if (row_idx < height_out && col_idx < width_out)
    {
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
    cudaMalloc((void **)&device_input, num_samples * input_channel * height_in * width_in * sizeof(float));              // input features map is input_channel
    cudaMalloc((void **)&device_output, num_samples * output_channel * height_out * width_out * sizeof(float));          // output feature map is output_channel
    cudaMalloc((void **)&device_weight, output_channel * input_channel * kernel_height * kernel_height * sizeof(float)); // input_channel * output_channel filter Maps of size kernel_height * kernel_height

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
