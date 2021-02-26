//
// Created by denis on 2/20/21.
//

#ifndef OPENVXFASTEXTRACTOR_IC_ANGLES_H
#define OPENVXFASTEXTRACTOR_IC_ANGLES_H

#include <vector>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include "../util.h"
#include "../user_extensions.h"
#include "VX/vx.h"

const int HALF_PATCH_SIZE = 15;

void IC_Angles(const cv::Mat &image, vx_keypoint_t *kp_buf, vx_size kp_size, vx_size kp_stride, vx_int32 *u_max_buf, vx_size u_max_size, vx_size u_max_stride);

vx_array createUMax(vx_context context);

extern "C" void IC_Angles_gpu(const cv::cuda::GpuMat &image,vx_keypoint_t *kp_buf, vx_size kp_size, vx_size kp_stride, vx_int32 *u_max_buf,
                              vx_size u_max_size, vx_size u_max_stride, cudaStream_t stream);

#include <stdio.h>

extern "C" {
#define ERROR_CHECK_STATUS( status ) { \
            vx_status status_ = (status); \
            if(status_ != VX_SUCCESS) { \
                printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
                exit(1); \
            } \
        }
static inline int localCeil( double value ) {
    int i = (int)value;
    return i + (i < value);
}
static inline int localFloor( double value ) {
    int i = (int)value;
    return i - (i > value);
}
static inline int localRound( int value ) {
    return value;
}
}

inline vx_array createUMax(vx_context context, vx_size u_max_size) {
    vx_array u_max = vxCreateArray(context, VX_TYPE_INT32, u_max_size + 1);
    std::vector<int> u_max_data(u_max_size + 1);

    int v, v0, vmax = localFloor(u_max_size * sqrt(2.f) / 2 + 1);
    int vmin = localCeil(u_max_size * sqrt(2.f) / 2);
    const double hp2 = u_max_size*u_max_size;


    for (v = 0; v <= vmax; ++v)
        u_max_data[v] = localRound(sqrt(hp2 - v * v));

    for (v = u_max_size, v0 = 0; v >= vmin; --v)
    {
        while (u_max_data[v0] == u_max_data[v0 + 1])
            ++v0;
        u_max_data[v] = v0;
        ++v0;
    }
    ERROR_CHECK_STATUS(vxAddArrayItems(u_max, u_max_size + 1, u_max_data.data(), sizeof(vx_int32)));

    return u_max;
}
#endif //OPENVXFASTEXTRACTOR_IC_ANGLES_H
