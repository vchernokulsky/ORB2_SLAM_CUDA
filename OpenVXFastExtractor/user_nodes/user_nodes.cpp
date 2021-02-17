//
// Created by denis on 16.02.2021.
//

#include "user_nodes.hpp"

extern "C" {
    enum user_library_e {
        USER_LIBRARY = 1,
    };
    enum user_kernel_e {
        USER_KERNEL_IC_ANGLE = VX_KERNEL_BASE(VX_ID_DEFAULT, USER_LIBRARY) + 0x004
    };

    vx_status registerUserNodes(vx_context context) {
        return registerIC_Angles_kernel(context);
    }

    vx_node IC_AnglesNode(vx_graph graph, vx_image image, vx_array corners, vx_array u_max) {
        vx_context context = vxGetContext((vx_reference) graph);
        vx_kernel kernel = vxGetKernelByEnum(context, USER_KERNEL_IC_ANGLE);
        ERROR_CHECK_OBJECT(kernel);
        vx_node node = vxCreateGenericNode(graph, kernel);
        ERROR_CHECK_OBJECT(node);
        ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 0, (vx_reference) image));
        ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 1, (vx_reference) corners));
        ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 2, (vx_reference) u_max));
        ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
    }

    vx_status VX_CALLBACK
    IC_Angles_validator(vx_node node, vx_reference const *parameters, vx_uint32 num, vx_meta_format *metas) {
        return VX_SUCCESS;
    }

    vx_status registerIC_Angles_kernel(vx_context context) {
        vx_kernel kernel = vxAddUserKernel(context,
                                           "user.kernel.IC_Angle",
                                           USER_KERNEL_IC_ANGLE,
                                           IC_Angles_function,
                                           3,
                                           IC_Angles_validator,
                                           NULL,
                                           NULL);
        ERROR_CHECK_OBJECT(kernel);

        ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
        ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
        vxAddLogEntry((vx_reference) context, VX_FAILURE, "user.kernel.IC_Angles\n");
        return VX_SUCCESS;
    }
}

vx_status VX_CALLBACK IC_Angles_function(vx_node node, const vx_reference *refs, vx_uint32 num) {
    vx_image vxImage = (vx_image) refs[0];
    vx_array vxKpArray = (vx_array) refs[1];
    vx_array u_max = (vx_array) refs[2];

    vx_size kp_size = 0;
    vxQueryArray(vxKpArray, VX_ARRAY_NUMITEMS, &kp_size, sizeof(kp_size));

    vx_size kp_stride;
    vx_map_id kp_map_id;
    vx_keypoint_t *kp_buf;
    ERROR_CHECK_STATUS(vxMapArrayRange(vxKpArray, 0, kp_size, &kp_map_id, &kp_stride, (void **) &kp_buf, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0));

    vx_uint32 vxImageWidth = 0, vxImageHeight = 0;
    ERROR_CHECK_STATUS(vxQueryImage(vxImage, VX_IMAGE_WIDTH, &vxImageWidth, sizeof(vxImageWidth)));
    ERROR_CHECK_STATUS(vxQueryImage(vxImage, VX_IMAGE_HEIGHT, &vxImageHeight, sizeof(vxImageHeight)));

    vx_rectangle_t image_rect = {0, 0, vxImageWidth, vxImageHeight};
    vx_map_id image_map_id;
    vx_imagepatch_addressing_t image_addr;
    void *image_data_ptr;
    ERROR_CHECK_STATUS(vxMapImagePatch(vxImage, &image_rect, 0, &image_map_id, &image_addr, &image_data_ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));

    const cv::Mat cvImage(vxImageHeight, vxImageWidth, CV_8U, image_data_ptr,  image_addr .stride_y );

    vx_size u_max_size = 0;
    vxQueryArray(u_max, VX_ARRAY_NUMITEMS, &u_max_size, sizeof(u_max_size));

    vx_size u_max_stride;
    vx_map_id u_max_map_id;
    int32_t *u_max_buf;
    ERROR_CHECK_STATUS(vxMapArrayRange(u_max, 0, u_max_size, &u_max_map_id, &u_max_stride, (void **) &u_max_buf, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0));

    IC_Angles(cvImage,kp_buf, kp_size, kp_stride, u_max_buf);
}