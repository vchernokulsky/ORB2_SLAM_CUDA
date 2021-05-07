//
// Created by denis on 16.02.2021.
//
#include <opencv2/opencv.hpp>
#include "../IC_Angles.h"

constexpr int u_max[] = {15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3};

void
IC_Angles(const cv::Mat &image, vx_keypoint_t *kp_buf, vx_size kp_size, vx_size kp_stride)
{
    int m_01, m_10;
    int step;
    int v_sum;
    int d;
    int val_plus;
    int val_minus;
    int v;
    int u;
    vx_keypoint_t kp;

    for(vx_size kp_i = 0; kp_i < kp_size; kp_i++) {
        m_01 = 0, m_10 = 0;
        kp = vxArrayItem(vx_keypoint_t, kp_buf, kp_i, kp_stride);

        if (kp.x > 15 && kp.x < image.size().width - 15 && kp.y > 15 && kp.y < image.size().height - 15) {
            const uchar *center = &image.at<uchar>(cvRound(kp.y), cvRound(kp.x));
            // Treat the center line differently, v=0
            for (u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
                m_10 += u * center[u];

            // Go line by line in the circular patch
            step = (int) image.step1();
            for (v = 1; v <= HALF_PATCH_SIZE; ++v) {
                // Proceed over the two lines
                v_sum = 0;
                d = u_max[v];
                for (u = -d; u <= d; ++u) {
                    val_plus = center[u + v * step];
                    val_minus = center[u - v * step];
                    v_sum += (val_plus - val_minus);
                    m_10 += u * (val_plus + val_minus);
                }
                m_01 += v * v_sum;
            }
        }

        vxArrayItem(vx_keypoint_t, kp_buf, kp_i, kp_stride).orientation = atan_f32((float)m_01, (float)m_10);
    }
}