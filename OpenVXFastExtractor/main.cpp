#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "opencv_util/opencv_interop.h"
#include "user_extensions/user_extensions.h"

#include "VX/vx.h"
#include "NVX/nvx.h"
#include "NVX/nvxcu.h"

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
    vx_graph graph = vxCreateGraph(context);

    ERROR_CHECK_STATUS(registerUserExtensions(context));

    cv::VideoCapture camera(0);

    double cameraWidth = 640;
    double cameraHeight = 480;

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
    vx_float32 fast_strength_thresh  = 20.0f;
    vx_size num_corners_value;
    std::vector<vx_array> fastCorners(levelsNum);
    {
        vx_size vxArraySize = 30 * 1000;
        fastCorners.at(0) = vxCreateArray(context, VX_TYPE_KEYPOINT, vxArraySize);
        for (uint8_t i = 1; i < levelsNum; ++i) {
            vxArraySize *= scaleFactor;
            fastCorners.at(i) = vxCreateArray(context, VX_TYPE_KEYPOINT, vxArraySize);
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
        //fastCornersNodes.at(i) = vxFastCornersNode(graph,
        //                                         vxPyramidImages.at(i),
        //                                       strength_thresh.at(i),
        //                                     vx_true_e,
        //                                   fastCorners.at(i),
        //                                 num_corners.at(i));
        fastCornersNodes.at(i) = nvxFastTrackNode(graph, vxPyramidImages.at(i), fastCorners.at(i), nullptr, nullptr, 9, fast_strength_thresh, 6, num_corners.at(i));
    }

    vx_array u_max = createUMax(context, HALF_PATCH_SIZE);
    std::vector<vx_array> IC_AnglesCorners(levelsNum);
    {
        vx_size vxArraySize = 30 * 1000;
        IC_AnglesCorners.at(0) = vxCreateArray(context, VX_TYPE_KEYPOINT, vxArraySize);
        for (uint8_t i = 1; i < levelsNum; ++i) {
            vxArraySize *= scaleFactor;
            IC_AnglesCorners.at(i) = vxCreateArray(context, VX_TYPE_KEYPOINT, vxArraySize);
        }
    }
    std::vector<vx_node> IC_AnglesNodes(levelsNum);
    for(uint8_t i = 0; i < levelsNum; ++i) {
        IC_AnglesNodes.at(i) = IC_AnglesNode(graph, vxPyramidImages.at(i), fastCorners.at(i), u_max, IC_AnglesCorners.at(i));
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

    vxSetGraphAttribute(graph, NVX_GRAPH_VERIFY_OPTIONS, "-O3", 3);

    ERROR_CHECK_STATUS(vxVerifyGraph(graph));

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

    cv::Mat tmp;
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start).count() <= 60) {
        rect.start_x = 0;
        rect.start_y = 0;
        rect.end_x = cameraWidth;
        rect.end_y = cameraHeight;

        vxMapImagePatch(vxInputImg, &rect, 0, &vxImageMapId, &vxImageAddr, &vxImagePtr, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);
        camera >> cvPreInputImg;
        cv::resize(cvPreInputImg, cvPreInputImg, cv::Size(640,480), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(cvPreInputImg, tmp, cv::COLOR_BGR2GRAY);
        tmp.copyTo(cvInputImg);
        vxUnmapImagePatch(vxInputImg, vxImageMapId);
        auto s = std::chrono::system_clock::now();
        vxProcessGraph(graph);
        std::cout << "t: " << (((float)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - s).count()) / (float)1000) << std::endl;
        for(uint8_t i = 0; i < levelsNum; ++i) {
            vx_size corners_count = 0;
            vxQueryArray(fastCorners.at(i), VX_ARRAY_NUMITEMS, &corners_count, sizeof( corners_count));

            vxQueryImage(gaussian7x7Images.at(i), VX_IMAGE_WIDTH, (void *)&(rect.end_x), size);
            vxQueryImage(gaussian7x7Images.at(i), VX_IMAGE_HEIGHT, (void *)&(rect.end_y), size);

            vxMapImagePatch(gaussian7x7Images.at(i), &rect, 0, &vxImageMapId, &vxImageAddr, &vxImagePtr, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);

            std::cout << "(" << (int)i << ") Fast corners: " << corners_count << std::endl;
            if (corners_count > 0) {
                vx_size kp_stride;
                vx_map_id kp_map;
                vx_uint8 * kp_buf;
                ERROR_CHECK_STATUS( vxMapArrayRange(fastCorners.at(i), 0, corners_count, &kp_map, &kp_stride, ( void ** ) &kp_buf, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0 ) );
                for( vx_size j = 0; j < corners_count; j++ ) {
                    vx_keypoint_t kp = vxArrayItem(vx_keypoint_t, kp_buf, j, kp_stride);
                    if( kp.tracking_status ) {
                        drawPoint(cvOutputImages.at(i), kp.x, kp.y);
                    }
                }
                ERROR_CHECK_STATUS( vxUnmapArrayRange(fastCorners.at(i), kp_map ) );
            }

            vx_size IC_AnglesCornersSize = 0;
            vxQueryArray(IC_AnglesCorners.at(i), VX_ARRAY_NUMITEMS, &IC_AnglesCornersSize, sizeof( IC_AnglesCornersSize));
            std::cout << "(" << (int)i << ") IC_Angles corners: " << IC_AnglesCornersSize << std::endl;

            cv::imshow("Webcam " + std::to_string(i), cvOutputImages.at(i));

            if (cv::waitKey(1) == 'q') {
                break;
            }
            vxUnmapImagePatch(gaussian7x7Images.at(i), vxImageMapId);
        }

        framesNum++;
    }

    std::cout << "framesNum: " << framesNum << std::endl;

    vxReleaseGraph(&graph);
    vxReleaseContext(&context);
    return 0;
}
