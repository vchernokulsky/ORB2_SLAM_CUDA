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

void IC_Angles(const cv::Mat &image, vx_keypoint_t *kp_buf, vx_size kp_size, vx_size kp_stride);

extern "C" void IC_Angles_gpu(const cv::cuda::GpuMat &image, vx_keypoint_t *kp_buf, vx_size kp_size, vx_size kp_stride,
                              cudaStream_t stream);

#include <stdio.h>

extern "C" {
#define ERROR_CHECK_STATUS( status ) { \
            vx_status status_ = (status); \
            if(status_ != VX_SUCCESS) { \
                printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
                exit(1); \
            } \
        }
}

constexpr int u_max[] = {15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3};

#endif //OPENVXFASTEXTRACTOR_IC_ANGLES_H
