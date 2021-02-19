
#include "opencv_interop.h"

void drawPoint(cv::Mat mat, int x, int y ) {
    cv::Point  center( x, y );
    cv::circle( mat, center, 1, cv::Scalar( 0, 0, 2), 2 );
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
     return 0;
 }

int convertVXMatrixTypeToCVMatType(vx_enum matrix_type)
{
    switch(matrix_type)
    {
    case VX_TYPE_UINT8:
        return CV_8UC1;
    case VX_TYPE_INT8:
        return CV_8SC1;
    case VX_TYPE_UINT16:
        return CV_16UC1;
    case VX_TYPE_INT16:
        return CV_16SC1;
    case VX_TYPE_UINT32:
    case VX_TYPE_INT32:
        return CV_32SC1;
    case VX_TYPE_FLOAT32:
        return CV_32FC1;
    case VX_TYPE_FLOAT64:
        return CV_64FC1;
    }
    return 0;
}

void copyVXMatrixToCVMat(vx_matrix src_mat, cv::Mat& dst_mat)
{
    vx_status status = VX_SUCCESS;
    vx_size rows_num = 0, cols_num = 0;
    vx_enum elem_type = 0;
    status = vxQueryMatrix(src_mat, VX_MATRIX_ATTRIBUTE_TYPE, &elem_type, sizeof(elem_type) );
    CV_Assert(status == VX_SUCCESS);
    status = vxQueryMatrix(src_mat, VX_MATRIX_ATTRIBUTE_ROWS, &rows_num, sizeof(rows_num) );
    CV_Assert(status == VX_SUCCESS);
    CV_Assert(rows_num <= static_cast<vx_size>(std::numeric_limits<int>::max()));
    status = vxQueryMatrix(src_mat, VX_MATRIX_ATTRIBUTE_COLUMNS, &cols_num, sizeof(cols_num) );
    CV_Assert(status == VX_SUCCESS);
    CV_Assert(cols_num <= static_cast<vx_size>(std::numeric_limits<int>::max()));
    dst_mat.create(static_cast<int>(rows_num), static_cast<int>(cols_num), convertVXMatrixTypeToCVMatType(elem_type));
    CV_Assert(dst_mat.isContinuous());
    status = vxReadMatrix(src_mat, dst_mat.data);
    CV_Assert(status == VX_SUCCESS);
}
void copyCVMatToVXMatrix(const cv::Mat& src_mat, vx_matrix dst_mat)
{
    vx_status status = VX_SUCCESS;
    vx_size dst_rows_num = 0, dst_cols_num = 0;
    vx_enum dst_elem_type = 0;
    status = vxQueryMatrix(dst_mat, VX_MATRIX_ATTRIBUTE_TYPE, &dst_elem_type, sizeof(dst_elem_type) );
    CV_Assert(status == VX_SUCCESS);
    status = vxQueryMatrix(dst_mat, VX_MATRIX_ATTRIBUTE_ROWS, &dst_rows_num, sizeof(dst_rows_num) );
    CV_Assert(status == VX_SUCCESS);
    status = vxQueryMatrix(dst_mat, VX_MATRIX_ATTRIBUTE_COLUMNS, &dst_cols_num, sizeof(dst_cols_num) );
    CV_Assert(status == VX_SUCCESS);
    CV_Assert(src_mat.isContinuous());
    CV_Assert(static_cast<vx_size>(src_mat.cols) == dst_cols_num && static_cast<vx_size>(src_mat.rows) == dst_rows_num);
    CV_Assert(src_mat.type() == convertVXMatrixTypeToCVMatType(dst_elem_type));
    status = vxWriteMatrix(dst_mat, src_mat.data);
    CV_Assert(status == VX_SUCCESS);
}
vx_matrix cloneCVMatToVXMatrix(vx_context context, const cv::Mat& src_mat)
{
    vx_matrix res_matrix = vxCreateMatrix(context, convertCVMatTypeToVXMatrixType(src_mat.type()),
                                                                                        src_mat.cols, src_mat.rows);
    CV_Assert(vxGetStatus((vx_reference)res_matrix) == VX_SUCCESS);
    copyCVMatToVXMatrix(src_mat, res_matrix);
    return res_matrix;
}

inline vx_enum convertCVMatTypeToVXMatrixType(int mat_type)
{
    switch(mat_type)
    {
    case CV_8UC1:
        return VX_TYPE_UINT8;
    case CV_8SC1:
        return VX_TYPE_INT8;
    case CV_16UC1:
        return VX_TYPE_UINT16;
    case CV_16SC1:
        return VX_TYPE_INT16;
    case CV_32SC1:
        return VX_TYPE_INT32;
    case CV_32FC1:
        return VX_TYPE_FLOAT32;
    case CV_64FC1:
        return VX_TYPE_FLOAT64;
    }
    return 0;
}