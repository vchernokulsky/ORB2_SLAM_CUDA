//
// Created by denis on 14.02.2021.
//

#ifndef OPENVXFASTEXTRACTOR_OPENCV_INTEROP_H
#define OPENVXFASTEXTRACTOR_OPENCV_INTEROP_H

#include <opencv2/opencv.hpp>
#include "VX/vx.h"

void drawPoint(cv::Mat mat, int x, int y );
vx_image createVXImageFromCVMat(vx_context context, const cv::Mat& mat);
vx_df_image convertCVMatTypeToVXImageFormat(int mat_type);
int convertVXImageFormatToCVMatType(vx_df_image format, vx_uint32 plane_index = 0);

#endif //OPENVXFASTEXTRACTOR_OPENCV_INTEROP_H
