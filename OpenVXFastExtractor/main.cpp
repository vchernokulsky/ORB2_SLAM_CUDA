#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "opencv_util/opencv_interop.h"
#include "user_nodes/user_nodes.hpp"

#include "VX/vx.h"
#include<unistd.h>

#include <chrono>
using namespace cv;

#define ERROR_CHECK_STATUS( status ) { \
        vx_status status_ = (status); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
    }


int main(void) {
    vx_context context = vxCreateContext();

    ERROR_CHECK_STATUS(registerUserNodes(context));

    vx_graph graph = vxCreateGraph(context);

    cv::VideoCapture camera(0);

    camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    double cameraWidth = camera.get(cv::CAP_PROP_FRAME_WIDTH);
    double cameraHeight = camera.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::cout << "cameraWidth: " << cameraWidth << std::endl;
    std::cout << "cameraHeight: " << cameraHeight << std::endl;

    usleep(3 * 1000 * 1000);

    size_t levelsNum = 8;
    double scaleFactor = 1 / 1.2;

    cv::Mat cvInputImg(cameraHeight, cameraWidth, CV_8UC1);
    vx_image vxInputImg = createVXImageFromCVMat(context, cvInputImg);

    std::vector<vx_image> vxPyramidImages{vxInputImg};
    vxPyramidImages.resize(levelsNum);
    {
        double scaleImageWidth = cameraWidth;
        double scaleImageHeight = cameraHeight;
        for (uint8_t i = 1; i < levelsNum; ++i) {
            scaleImageWidth *= scaleFactor;
            scaleImageHeight *= scaleFactor;

            vxPyramidImages.at(i) = vxCreateVirtualImage(graph, scaleImageWidth, scaleImageHeight, VX_DF_IMAGE_U8);
        }
    }
    std::vector<vx_node> scaleNodes(levelsNum - 1);
    for(uint8_t i = 1; i < levelsNum; ++i) {
        scaleNodes.at(i - 1) = vxScaleImageNode(graph, vxPyramidImages.at(i - 1), vxPyramidImages.at(i), VX_INTERPOLATION_BILINEAR);
    }

    vx_float32 fast_strength_thresh  = 7.0f;
    vx_size num_corners_value;
    std::vector<vx_array> vxCorners(levelsNum);
    {
        uint16_t vxArraySize = 30 * 1000;
        vxCorners.at(0) = vxCreateArray(context, VX_TYPE_KEYPOINT, vxArraySize);
        for (uint8_t i = 1; i < levelsNum; ++i) {
            vxArraySize *= scaleFactor;
            vxCorners.at(i) = vxCreateArray(context, VX_TYPE_KEYPOINT, vxArraySize);
        }
    }
    std::vector<vx_scalar> strength_thresh(levelsNum);
    for(uint8_t i = 0; i < levelsNum; ++i) {
        strength_thresh.at(i) = vxCreateScalar( context, VX_TYPE_FLOAT32, &fast_strength_thresh );
    }
    std::vector<vx_scalar> num_corners(levelsNum);
    for(uint8_t i = 0; i < levelsNum; ++i) {
        num_corners.at(i) = vxCreateScalar( context, VX_TYPE_SIZE, &num_corners_value );
    }
    std::vector<vx_node> fastCornersNodes(levelsNum);
    for(uint8_t i = 0; i < levelsNum; ++i) {
        fastCornersNodes.at(i) = vxFastCornersNode(graph,
                                                   vxPyramidImages.at(i),
                                                   strength_thresh.at(i),
                                                   vx_true_e,
                                                   vxCorners.at(i),
                                                   num_corners.at(i));
    }

    std::vector<cv::Mat> cvOutputImages(levelsNum);
    {
        double outputImageWidth = cameraWidth, outputImageHeight = cameraHeight;
        cvOutputImages.at(0) = cvInputImg;
        for (uint8_t i = 1; i < levelsNum; ++i) {
            outputImageWidth *= scaleFactor;
            outputImageHeight *= scaleFactor;

            cvOutputImages.at(i) = cv::Mat(outputImageHeight, outputImageWidth, CV_8UC1);
        }
    }
    std::vector<vx_image> gaussian7x7Images(levelsNum);
    {
        double scaleImageWidth = cameraWidth;
        double scaleImageHeight = cameraHeight;
        gaussian7x7Images.at(0) = createVXImageFromCVMat(context, cvOutputImages.at(0));
        for (uint8_t i = 0; i < levelsNum; ++i) {
            scaleImageWidth *= scaleFactor;
            scaleImageHeight *= scaleFactor;

            gaussian7x7Images.at(i) = createVXImageFromCVMat(context, cvOutputImages.at(i));
        }
    }

    vx_int32 gaussian7x7Coefs[7][7] = {
            {1, 2,  3,  4,  3, 2, 1},
            {2, 4,  6,  7,  6, 4, 2},
            {3, 6,  9, 10,  9, 6, 3},
            {4, 7, 10, 12, 10, 7, 4},
            {3, 6,  9, 10,  9, 6, 3},
            {2, 4,  6,  7,  6, 4, 2},
            {1, 2,  3,  4,  3, 2, 1},
    };

    vx_convolution gaussian7x7 = vxCreateConvolution(context, 7, 7);
    vxCopyConvolutionCoefficients(gaussian7x7, (vx_int32*)gaussian7x7Coefs,
                                  VX_WRITE_ONLY,
                                  VX_MEMORY_TYPE_HOST);
    vx_uint32 scale = 256;
    vxSetConvolutionAttribute(gaussian7x7, VX_CONVOLUTION_SCALE, &scale, sizeof(scale));
    std::vector<vx_node> gaussian7x7Nodes(levelsNum);
    for(uint8_t i = 0; i < levelsNum; ++i) {
        gaussian7x7Nodes.at(i) = vxConvolveNode(graph, vxPyramidImages.at(i), gaussian7x7, gaussian7x7Images.at(i));
    }

    vx_status status = vxVerifyGraph(graph);

    if (VX_SUCCESS != status) {
        std::cout << "invalid graph" << std::endl;
        return 1;
    }

    std::cout << "graph is valid" << std::endl;

    for(auto& e: num_corners) {
        vxReleaseScalar(&e);
    }
    for(auto& e: strength_thresh) {
        vxReleaseScalar(&e);
    }

    if (!camera.isOpened()) {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return 1;
    }

    vx_imagepatch_addressing_t vxImageAddr;
    void *vxImagePtr;
    vx_map_id vxImageMapId;
    vx_rectangle_t rect;
    vx_size size = sizeof(rect.end_x);

    cv::Mat cvPreInputImg;

    long long int framesNum = 0;
    auto start = std::chrono::system_clock::now();

    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start).count() <= 60) {
        rect.start_x = 0;
        rect.start_y = 0;
        rect.end_x = cameraWidth;
        rect.end_y = cameraHeight;

        vxMapImagePatch(vxInputImg, &rect, 0, &vxImageMapId, &vxImageAddr, &vxImagePtr, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);
        camera >> cvPreInputImg;
        cv::cvtColor(cvPreInputImg, cvInputImg, CV_BGR2GRAY);
        vxUnmapImagePatch(vxInputImg, vxImageMapId);

        vxProcessGraph(graph);

        for(uint8_t i = 0; i < levelsNum; ++i) {
            vx_size corners_count = 0;
            vxQueryArray( vxCorners.at(i), VX_ARRAY_NUMITEMS, &corners_count, sizeof( corners_count));

            vxQueryImage(vxPyramidImages.at(i), VX_IMAGE_WIDTH, (void *)&(rect.end_x), size);
            vxQueryImage(vxPyramidImages.at(i), VX_IMAGE_HEIGHT, (void *)&(rect.end_y), size);

            vxMapImagePatch(vxPyramidImages.at(i), &rect, 0, &vxImageMapId, &vxImageAddr, &vxImagePtr, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);
            std::cout << "(" << (int)i << ") num_corners: " << corners_count << std::endl;
            if (corners_count > 0) {
                vx_size kp_stride;
                vx_map_id kp_map;
                vx_uint8 * kp_buf;
                ERROR_CHECK_STATUS( vxMapArrayRange(vxCorners.at(i), 0, corners_count, &kp_map, &kp_stride, ( void ** ) &kp_buf, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0 ) );
                for( vx_size j = 0; j < corners_count; j++ ) {
                    vx_keypoint_t * kp = (vx_keypoint_t *) (kp_buf + j * kp_stride );
                    if( kp->tracking_status ) {
                        drawPoint(cvOutputImages.at(i), kp->x, kp->y);
                    }
                }
                ERROR_CHECK_STATUS( vxUnmapArrayRange(vxCorners.at(i), kp_map ) );
            }

            cv::imshow("Webcam " + std::to_string(i), cvOutputImages.at(i));

            if (cv::waitKey(1) == 'q') {
                break;
            }
            vxUnmapImagePatch(vxPyramidImages.at(i), vxImageMapId);
        }

        framesNum++;
    }

    std::cout << "framesNum: " << framesNum << std::endl;

    vxReleaseGraph(&graph);
    vxReleaseContext(&context);
    return 0;
}