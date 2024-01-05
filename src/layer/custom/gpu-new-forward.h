#ifndef SRC_LAYER_CUSTOM_GPU_NEW_FORWARD_H
#define SRC_LAYER_CUSTOM_GPU_NEW_FORWARD_H
#include "./gpu-support.h"

class GPUInterface
{
    public:
    void get_device_properties();
    void conv_forward_gpu_full(float *host_y, const float *host_x, const float *host_k,
                                                  const int B, const int M, const int C,
                                                  const int H, const int W, const int K);
};

#endif