//
// Created by denis on 16.02.2021.
//

#include "user_extensions.h"

#include "NVX/nvx.h"
#include <cuda_runtime.h>
#include "opencv2/cudafeatures2d.hpp"

static vx_status VX_CALLBACK IC_Angles_cpu_function(vx_node node, const vx_reference * refs, vx_uint32 num);
static vx_status VX_CALLBACK IC_Angles_gpu_function(vx_node node, const vx_reference * refs, vx_uint32 num);

enum {
    USER_LIBRARY = 0x1,
    USER_KERNEL_IC_ANGLE_CPU = VX_KERNEL_BASE(VX_ID_DEFAULT, USER_LIBRARY) + 0x0,
    USER_KERNEL_IC_ANGLE_GPU = VX_KERNEL_BASE(VX_ID_DEFAULT, USER_LIBRARY) + 0x1,
};

static vx_status VX_CALLBACK IC_Angles_validator(vx_node node, vx_reference const *parameters, vx_uint32 num, vx_meta_format *metas);
static vx_status registerIC_Angles_kernels(vx_context context);

vx_status registerUserExtensions(vx_context context) {
    return registerIC_Angles_kernels(context);
}

vx_node
IC_AnglesNodeCpu(vx_graph graph, vx_image image, vx_array input_keypoints, vx_array output_keypoints) {
    vx_context context = vxGetContext((vx_reference) graph);
    vx_kernel kernel = vxGetKernelByEnum(context, USER_KERNEL_IC_ANGLE_CPU);
    ERROR_CHECK_OBJECT(kernel);
    vx_node node = vxCreateGenericNode(graph, kernel);
    ERROR_CHECK_OBJECT(node);
    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 0, (vx_reference) image));
    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 1, (vx_reference) input_keypoints));
    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 2, (vx_reference) output_keypoints));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
    return node;
}

vx_node
IC_AnglesNodeGpu(vx_graph graph, vx_image image, vx_array input_keypoints, vx_array output_keypoints) {
vx_context context = vxGetContext((vx_reference) graph);
vx_kernel kernel = vxGetKernelByEnum(context, USER_KERNEL_IC_ANGLE_GPU);
ERROR_CHECK_OBJECT(kernel);
vx_node node = vxCreateGenericNode(graph, kernel);
ERROR_CHECK_OBJECT(node);
ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 0, (vx_reference) image));
ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 1, (vx_reference) input_keypoints));
ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 2, (vx_reference) output_keypoints));
ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
return node;
}

vx_status VX_CALLBACK
IC_Angles_validator(vx_node node, vx_reference const *parameters, vx_uint32 num, vx_meta_format *metas) {
    if (num != 3)
    {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    vx_parameter param1 = vxGetParameterByIndex(node, 0);

    vx_df_image src_format = 0;
    vx_image    input_mage = 0;

    vxQueryParameter(param1, VX_PARAMETER_REF, &input_mage, sizeof(input_mage));

    vxQueryImage(input_mage, VX_IMAGE_FORMAT, &src_format, sizeof(src_format));

    if (src_format != VX_DF_IMAGE_U8) {
        return VX_ERROR_INVALID_TYPE;
    }

    vx_enum param_type;
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[1], VX_ARRAY_ITEMTYPE, &param_type, sizeof(param_type)));
    if(param_type != VX_TYPE_KEYPOINT)
    {
        vxAddLogEntry((vx_reference)node, VX_FAILURE, "Invalid array type for parameter 1");
        return VX_ERROR_INVALID_TYPE;
    }

    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[2], VX_ARRAY_ITEMTYPE, &param_type, sizeof(param_type)));
    if(param_type != VX_TYPE_KEYPOINT)
    {
        vxAddLogEntry((vx_reference)node, VX_FAILURE, "Invalid array type for parameter 2");
        return VX_ERROR_INVALID_TYPE;
    }

    ERROR_CHECK_STATUS(vxSetMetaFormatFromReference(metas[2], parameters[1]))

    return VX_SUCCESS;
}

vx_status registerIC_Angles_kernels(vx_context context) {
    vx_kernel kernel_gpu = vxAddUserKernel(context,
                                           "gpu:user.kernel.IC_AngleGpu",
                                           USER_KERNEL_IC_ANGLE_GPU,
                                           IC_Angles_gpu_function,
                                           3,
                                           IC_Angles_validator,
                                           NULL,
                                           NULL);

    ERROR_CHECK_OBJECT(kernel_gpu);

    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel_gpu, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel_gpu, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel_gpu, 2, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel_gpu));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel_gpu));

    vx_kernel kernel_cpu = vxAddUserKernel(context,
                                           "user.kernel.IC_AngleCpu",
                                           USER_KERNEL_IC_ANGLE_CPU,
                                           IC_Angles_cpu_function,
                                           3,
                                           IC_Angles_validator,
                                           NULL,
                                           NULL);
    ERROR_CHECK_OBJECT(kernel_cpu);

    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel_cpu, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel_cpu, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel_cpu, 2, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel_cpu));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel_cpu));
    return VX_SUCCESS;
}

vx_status VX_CALLBACK IC_Angles_cpu_function(vx_node node, const vx_reference *refs, vx_uint32 num) {
    vx_image vxImage = (vx_image) refs[0];
    vx_array vxInputKeyPoints = (vx_array) refs[1];
    vx_array vxOutputKeyPoints = (vx_array) refs[2];

    vxTruncateArray(vxOutputKeyPoints, 0);
    vx_size input_kp_size = 0;
    vxQueryArray(vxInputKeyPoints, VX_ARRAY_NUMITEMS, &input_kp_size, sizeof(input_kp_size));

    if (input_kp_size > 1) {
        vx_uint32 vxImageWidth = 0, vxImageHeight = 0;
        ERROR_CHECK_STATUS(vxQueryImage(vxImage, VX_IMAGE_WIDTH, &vxImageWidth, sizeof(vxImageWidth)));
        ERROR_CHECK_STATUS(vxQueryImage(vxImage, VX_IMAGE_HEIGHT, &vxImageHeight, sizeof(vxImageHeight)));

        vx_rectangle_t image_rect = {0, 0, vxImageWidth, vxImageHeight};
        vx_map_id image_map_id;
        vx_imagepatch_addressing_t image_addr;
        void *image_data_ptr;
        ERROR_CHECK_STATUS(
                vxMapImagePatch(vxImage, &image_rect, 0, &image_map_id, &image_addr, &image_data_ptr, VX_READ_ONLY,
                                VX_MEMORY_TYPE_HOST, VX_NOGAP_X));

        const cv::Mat cvImage(vxImageHeight, vxImageWidth, CV_8U, image_data_ptr, image_addr.stride_y);

        vx_size input_kp_stride;
        vx_map_id input_kp_map_id;
        vx_keypoint_t *input_kp_buf;
        ERROR_CHECK_STATUS(
                vxMapArrayRange(vxInputKeyPoints, 0, input_kp_size, &input_kp_map_id, &input_kp_stride, (void **) &input_kp_buf, VX_READ_ONLY,
                                VX_MEMORY_TYPE_HOST, 0));

        vxAddArrayItems(vxOutputKeyPoints, input_kp_size, input_kp_buf, input_kp_stride);

        vx_size output_kp_stride;
        vx_map_id output_kp_map_id;
        vx_keypoint_t *output_kp_buf;
        ERROR_CHECK_STATUS(
                vxMapArrayRange(vxOutputKeyPoints, 0, input_kp_size, &output_kp_map_id, &output_kp_stride, (void **) &output_kp_buf, VX_READ_AND_WRITE,
                                VX_MEMORY_TYPE_HOST, 0));

        IC_Angles(cvImage, output_kp_buf, input_kp_size, output_kp_stride);

        ERROR_CHECK_STATUS(vxUnmapImagePatch(vxImage, image_map_id));
        ERROR_CHECK_STATUS(vxUnmapArrayRange(vxInputKeyPoints, input_kp_map_id));
        ERROR_CHECK_STATUS(vxUnmapArrayRange(vxOutputKeyPoints, output_kp_map_id));
    }

    return VX_SUCCESS;
}

vx_status VX_CALLBACK IC_Angles_gpu_function(vx_node node, const vx_reference *refs, vx_uint32 num) {
    vx_image vxImage = (vx_image) refs[0];
    vx_array vxInputKeyPoints = (vx_array) refs[1];
    vx_array vxOutputKeyPoints = (vx_array) refs[2];

    vxTruncateArray(vxOutputKeyPoints, 0);

    vx_size input_kp_size = 0;
    vxQueryArray(vxInputKeyPoints, VX_ARRAY_NUMITEMS, &input_kp_size, sizeof(input_kp_size));

    if (input_kp_size > 0) {
        cudaStream_t stream = NULL;
        vxQueryNode(node, NVX_NODE_CUDA_STREAM, &stream, sizeof(stream));

        vx_uint32 vxImageWidth = 0, vxImageHeight = 0;
        ERROR_CHECK_STATUS(vxQueryImage(vxImage, VX_IMAGE_WIDTH, &vxImageWidth, sizeof(vxImageWidth)));
        ERROR_CHECK_STATUS(vxQueryImage(vxImage, VX_IMAGE_HEIGHT, &vxImageHeight, sizeof(vxImageHeight)));

        vx_rectangle_t image_rect = {0, 0, vxImageWidth, vxImageHeight};
        vx_map_id image_map_id;
        vx_imagepatch_addressing_t image_addr;
        void *image_data_ptr;
        ERROR_CHECK_STATUS(
                vxMapImagePatch(vxImage, &image_rect, 0, &image_map_id, &image_addr, &image_data_ptr, VX_READ_ONLY,
                                NVX_MEMORY_TYPE_CUDA, VX_NOGAP_X));

        const cv::cuda::GpuMat cvImage(vxImageHeight, vxImageWidth, CV_8U, image_data_ptr, image_addr.stride_y);

        vx_size input_kp_stride;
        vx_map_id input_kp_map_id;
        vx_keypoint_t *input_kp_buf;
        ERROR_CHECK_STATUS(
                vxMapArrayRange(vxInputKeyPoints, 0, input_kp_size, &input_kp_map_id, &input_kp_stride, (void **) &input_kp_buf, VX_READ_ONLY,
                                VX_MEMORY_TYPE_HOST, 0));

        vxAddArrayItems(vxOutputKeyPoints, input_kp_size, input_kp_buf, input_kp_stride);

        vx_size output_kp_size = 0;
        vxQueryArray(vxOutputKeyPoints, VX_ARRAY_NUMITEMS, &output_kp_size, sizeof(output_kp_size));
        vx_size output_kp_stride;
        vx_map_id output_kp_map_id;
        vx_keypoint_t *output_kp_buf;
        ERROR_CHECK_STATUS(
                vxMapArrayRange(vxOutputKeyPoints, 0, output_kp_size, &output_kp_map_id, &output_kp_stride, (void **) &output_kp_buf, VX_READ_AND_WRITE,
                                NVX_MEMORY_TYPE_CUDA, 0));

        IC_Angles_gpu(cvImage, output_kp_buf, input_kp_size, output_kp_stride, stream);

        ERROR_CHECK_STATUS(vxUnmapImagePatch(vxImage, image_map_id));
        ERROR_CHECK_STATUS(vxUnmapArrayRange(vxInputKeyPoints, input_kp_map_id));
        ERROR_CHECK_STATUS(vxUnmapArrayRange(vxOutputKeyPoints, output_kp_map_id));
    }


    return VX_SUCCESS;
}
