#include "./gpu-new-forward.h"
#include <cmath>
#include <iostream>

#define TILE_WIDTH_C1 16
#define TILE_WIDTH_C3 12

__constant__ float deviceMaskData[2400];


__global__ void conv_forward_kernel(float* output, const float* input, const float* kernel, 
                                    const int num_samples, const int output_channel, const int input_channel, 
                                    const int height, const int width, const int kernel_size)
{
    int TILE_WIDTH;    
    if (input_channel == 1){
        TILE_WIDTH = TILE_WIDTH_C1;
    }
    else{
        TILE_WIDTH = TILE_WIDTH_C3;
    }
    extern __shared__ float shared_input[];
    const int INPUT_TILE_WIDTH = TILE_WIDTH + kernel_size -1;
    
    const int H_out = height - kernel_size + 1;
    const int W_out = width - kernel_size + 1;


#define y4d(i3, i2, i1, i0) output[(i3) * (output_channel * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) input[(i3) * (input_channel * height * width) + (i2) * (height * width) + (i1) * (width) + i0]
#define k4d(i3, i2, i1, i0) deviceMaskData[(i3) * (input_channel * kernel_size * kernel_size) + (i2) * (kernel_size * kernel_size) + (i1) * (kernel_size) + i0]
#define sm3d(i2, i1, i0) shared_input[(i2) * (INPUT_TILE_WIDTH * INPUT_TILE_WIDTH) + (i1) * INPUT_TILE_WIDTH + i0]

    int W_grid = ceil(1.0*W_out / TILE_WIDTH); 
    int b = blockIdx.x;                 // batch number
    int m = blockIdx.y;                 // output feature
    
    int ty = threadIdx.y;              // thread ID in the current TILE  
    int tx = threadIdx.x;
    
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + ty; // row of the input image matrix
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + tx; // col of the input image matrix'

    int startOfTile_h = (blockIdx.z / W_grid) * TILE_WIDTH; // row of the input image matrix
    int startOfTile_w = (blockIdx.z % W_grid) * TILE_WIDTH; // col of the input image matrix'

    #pragma unroll
    for (int c = 0; c < input_channel; c++)
    {
        #pragma unroll
        for(int i = ty; i < INPUT_TILE_WIDTH; i += TILE_WIDTH)
        {
            #pragma unroll
            for(int j = tx; j < INPUT_TILE_WIDTH; j += TILE_WIDTH)
            {
                if (startOfTile_h + i < height && startOfTile_w + j < width)
                {
                    sm3d(c, i, j) = x4d(b, c, startOfTile_h + i, startOfTile_w + j);
                }
            }
        }
    }

    // Make sure all threads loaded data into shared memory
    __syncthreads();

    // compute only within bounds
    if ((h < H_out) && (w < W_out)) 
    {
        float accum = 0.0f;
        #pragma unroll
        for(int c=0; c<input_channel; c++)             // sum over all input features
        {
            #pragma unroll
            for(int p=0; p< kernel_size; p++)         // KxK filter 
                #pragma unroll
                for(int q=0; q< kernel_size; q++)
                    accum += sm3d(c, p+ty, q+tx) * k4d(m, c, p, q); 
        }
        y4d(b,m,h,w) = accum;
    } 
    
    #undef sm4d
    #undef y4d
    #undef x4d
    #undef k4d
}

__host__ void GPUInterface::conv_forward_gpu_full(float *output_data, const float *input_data, const float *weight_data,
                                                  const int num_samples, const int output_channel, const int input_channel,
                                                  const int height_in, const int width_in, const int kernel_height)
{
    // Set the tile width
    int TILE_WIDTH;    
    if (input_channel == 1){
        TILE_WIDTH = TILE_WIDTH_C1;
    }
    else{
        TILE_WIDTH = TILE_WIDTH_C3;
    }
    std::cout << ". Optimize 02:\n";

    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = height_in - kernel_height + 1;
    const int W_out = width_in - kernel_height + 1;

    int inputSize = num_samples * input_channel * height_in * width_in * sizeof(float);
    int outputSize = num_samples * output_channel * H_out * W_out * sizeof(float);
    int maskSize = output_channel * input_channel * kernel_height * kernel_height * sizeof(float);

    float *device_input, *device_output, *device_kernel;

    cudaMalloc((void **)&device_input, inputSize);
    cudaMalloc((void **)&device_output, outputSize);
    cudaMalloc((void **)&device_kernel, maskSize);

    cudaMemcpy(device_input, input_data, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceMaskData, weight_data, maskSize);

    dim3 numThreadsPerBlock, numBlocksInGrid;

    int Z = ceil(1.0 * H_out / TILE_WIDTH) * ceil(1.0 * W_out / TILE_WIDTH);
    numThreadsPerBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
    int shmem_size = input_channel * (TILE_WIDTH + kernel_height - 1) * (TILE_WIDTH + kernel_height - 1) * sizeof(float);
    numBlocksInGrid = dim3(num_samples, output_channel, Z);
    
    // Launch the kernel
    conv_forward_kernel<<<numBlocksInGrid, numThreadsPerBlock, shmem_size>>>(device_output, device_input, device_kernel, num_samples, output_channel, input_channel, height_in, width_in, kernel_height);

    // Copy the output back to host
    cudaMemcpy(output_data, device_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_kernel);

}
