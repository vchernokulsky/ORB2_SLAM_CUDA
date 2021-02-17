//
// Created by denis on 16.02.2021.
//

#ifndef OPENVXFASTEXTRACTOR_USER_NODES_HPP
#define OPENVXFASTEXTRACTOR_USER_NODES_HPP

#include <opencv2/core.hpp>

#include "kernels/cpu/IC_Angle.hpp"

extern "C" {
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

    static vx_status VX_CALLBACK IC_Angles_validator(vx_node node, vx_reference const *parameters, vx_uint32 num, vx_meta_format *metas);
    static vx_status registerIC_Angles_kernel(vx_context context);
    vx_node IC_AnglesNode(vx_graph graph, vx_image image, vx_array corners, vx_array u_max);
    vx_status registerUserNodes(vx_context context);
};

static vx_status VX_CALLBACK IC_Angles_function(vx_node node, const vx_reference * refs, vx_uint32 num);

#endif //OPENVXFASTEXTRACTOR_USER_NODES_HPP
