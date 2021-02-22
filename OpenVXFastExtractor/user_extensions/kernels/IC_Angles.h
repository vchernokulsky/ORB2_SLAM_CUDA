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

void loadUMax(const int* u_max, int count);
void IC_Angle_gpu(cv::cuda::GpuMat &image, cv::KeyPoint * keypoints, int npoints, cudaStream_t stream);

#endif //OPENVXFASTEXTRACTOR_IC_ANGLES_H
