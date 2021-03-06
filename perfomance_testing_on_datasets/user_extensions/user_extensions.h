//
// Created by denis on 16.02.2021.
//

#ifndef OPENVXFASTEXTRACTOR_USER_EXTENSIONS_H
#define OPENVXFASTEXTRACTOR_USER_EXTENSIONS_H

#include <opencv2/core.hpp>
#include "kernels/IC_Angles.h"
#include "VX/vx.h"

#define ERROR_CHECK_STATUS( status ) { \
        vx_status status_ = (status); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
    }

#define ERROR_CHECK_OBJECT( obj ) { \
        vx_status status_ = vxGetStatus((vx_reference)(obj)); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
    }

vx_status registerUserExtensions(vx_context context);

vx_node
IC_AnglesNodeCpu(vx_graph graph, vx_image image, vx_array input_keypoints, vx_array output_keypoints);
vx_node
IC_AnglesNodeGpu(vx_graph graph, vx_image image, vx_array input_keypoints, vx_array output_keypoints);

#endif //OPENVXFASTEXTRACTOR_USER_EXTENSIONS_H
