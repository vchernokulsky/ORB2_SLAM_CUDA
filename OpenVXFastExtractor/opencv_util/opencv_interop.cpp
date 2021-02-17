
#include "opencv_interop.h"

void drawPoint(cv::Mat mat, int x, int y ) {
    cv::Point  center( x, y );
    cv::circle( mat, center, 1, cv::Scalar( 0, 0, 255 ), 2 );
}

vx_image createVXImageFromCVMat(vx_context context, const cv::Mat& mat) {
    vx_df_image format = convertCVMatTypeToVXImageFormat(mat.type());

    vx_imagepatch_addressing_t addrs[1];
    addrs[0].dim_x = mat.cols;
    addrs[0].dim_y = mat.rows;
    addrs[0].stride_x = static_cast<vx_int32>(mat.elemSize());
    addrs[0].stride_y = static_cast<vx_int32>(mat.step);

    void *ptrs[1] = { const_cast<uchar *>(mat.ptr()) };

    vx_image img = vxCreateImageFromHandle(context, format, addrs, ptrs, VX_MEMORY_TYPE_HOST);
    CV_Assert(vxGetStatus((vx_reference)img) == VX_SUCCESS);

    return img;
}

vx_df_image convertCVMatTypeToVXImageFormat(int mat_type)
{
    switch (mat_type)
    {
        case CV_8UC1:
            return VX_DF_IMAGE_U8;
        case CV_16UC1:
            return VX_DF_IMAGE_U16;
        case CV_16SC1:
            return VX_DF_IMAGE_S16;
        case CV_32SC1:
            return VX_DF_IMAGE_S32;
        case CV_8UC3:
            return VX_DF_IMAGE_RGB;
        case CV_8UC4:
            return VX_DF_IMAGE_RGBX;
    }
    CV_Error(CV_StsUnsupportedFormat, "Unsupported format");
    return 0;
}

int convertVXImageFormatToCVMatType(vx_df_image format, vx_uint32 plane_index)
 {
     switch (format)
     {
     case VX_DF_IMAGE_U8:
     case VX_DF_IMAGE_YUV4:
     case VX_DF_IMAGE_IYUV:
         return CV_8UC1;
     case VX_DF_IMAGE_U16:
         return CV_16UC1;
     case VX_DF_IMAGE_S16:
         return CV_16SC1;
     case VX_DF_IMAGE_U32:
     case VX_DF_IMAGE_S32:
         return CV_32SC1;
     case VX_DF_IMAGE_UYVY:
     case VX_DF_IMAGE_YUYV:
         return CV_8UC2;
     case VX_DF_IMAGE_RGB:
         return CV_8UC3;
     case VX_DF_IMAGE_RGBX:
         return CV_8UC4;
     case VX_DF_IMAGE_NV12:
     case VX_DF_IMAGE_NV21:
         return plane_index == 0 ? CV_8UC1 : CV_8UC2;
     }
     CV_Error(CV_StsUnsupportedFormat, "Unsupported format");
     return 0;
 }