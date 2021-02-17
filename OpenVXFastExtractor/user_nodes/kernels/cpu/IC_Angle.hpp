//
// Created by denis on 16.02.2021.
//

#ifndef OPENVXFASTEXTRACTOR_IC_ANGLE_HPP
#define OPENVXFASTEXTRACTOR_IC_ANGLE_HPP

#include <VX/vx.h>
#include <stdlib.h>
#include <opencv2/core.hpp>

const int HALF_PATCH_SIZE = 15;

void IC_Angles(const cv::Mat& image, vx_keypoint_t *kp_buf, vx_size kp_size, vx_size kp_stride, const vx_int32 *u_max_buf);

#endif //OPENVXFASTEXTRACTOR_IC_ANGLE_HPP
