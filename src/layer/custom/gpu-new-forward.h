#ifndef SRC_LAYER_CUSTOM_GPU_NEW_FORWARD_H
#define SRC_LAYER_CUSTOM_GPU_NEW_FORWARD_H

class GPUInterface
{
    public:
    void get_device_properties();
    void conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K);
    void conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K);
    void conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K);

    /* For creating a dummy kernel call so that we can distinguish between kernels launched for different layers
     * in the Nsight Compute CLI for measuring per layer Op Times
     */
    void insert_post_barrier_kernel();
    // For inserting a marker visible in Nsys so that we can time total student function time
    void insert_pre_barrier_kernel();
};

#endif