//
// Created by denis on 5/9/21.
//

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
    uint8_t borderWidth = 50;

    std::cout << "cameraWidth: " << cameraWidth << std::endl;
    std::cout << "cameraHeight: " << cameraHeight << std::endl;

    usleep(3 * 1000 * 1000);

    cv::Mat cvInputImg(cameraHeight, cameraWidth, CV_8U);
    vx_image vxInputImg = createVXImageFromCVMat(context, cvInputImg);

    cv::Mat cvOutputImg(cameraHeight + borderWidth * 2, cameraWidth  + borderWidth * 2, CV_8U);
    vx_image vxOutputImg = createVXImageFromCVMat(context, cvOutputImg);

    vx_scalar vxBorderWidth = vxCreateScalar( context, VX_TYPE_UINT8, &borderWidth );
    vx_node n = copyMakeBorderCpuNode(graph, vxInputImg, vxOutputImg, vxBorderWidth);

    vxSetGraphAttribute(graph, NVX_GRAPH_VERIFY_OPTIONS, "-O3", 3);

    ERROR_CHECK_STATUS(vxVerifyGraph(graph));

    std::cout << "graph is valid" << std::endl;

    if (!camera.isOpened()) {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return 1;
    }

    vx_imagepatch_addressing_t vxImageAddr;
    void *vxImagePtr;
    vx_map_id vxImageMapId;
    vx_rectangle_t rect;
    rect.start_x = 0;
    rect.start_y = 0;
    rect.end_x = cameraWidth;
    rect.end_y = cameraHeight;

    vx_imagepatch_addressing_t vxImageAddr1;
    void *vxImagePtr1;
    vx_map_id vxImageMapId1;
    vx_rectangle_t rect1;
    rect1.start_x = 0;
    rect1.start_y = 0;
    rect1.end_x = cameraWidth + borderWidth * 2;
    rect1.end_y = cameraHeight + borderWidth * 2;

    auto start = std::chrono::system_clock::now();

    cv::Mat tmp;
    while (true) {

        ERROR_CHECK_STATUS(vxMapImagePatch(vxInputImg, &rect, 0, &vxImageMapId, &vxImageAddr, &vxImagePtr, VX_READ_AND_WRITE,
                        VX_MEMORY_TYPE_HOST, 0));
        camera >> tmp;
        cv::resize(tmp, tmp, cv::Size(640,480), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);
        tmp.copyTo(cvInputImg);
        ERROR_CHECK_STATUS(vxUnmapImagePatch(vxInputImg, vxImageMapId));

        vxProcessGraph(graph);

        ERROR_CHECK_STATUS(vxMapImagePatch(vxOutputImg, &rect1, 0, &vxImageMapId1, &vxImageAddr1, &vxImagePtr1, VX_READ_ONLY,
                        VX_MEMORY_TYPE_HOST, 0));
        cv::imshow("border", cvOutputImg);
        cv::waitKey(1);
        ERROR_CHECK_STATUS(vxUnmapImagePatch(vxOutputImg, vxImageMapId1));
    }
    return 0;
}