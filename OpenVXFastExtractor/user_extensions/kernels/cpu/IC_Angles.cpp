//
// Created by denis on 16.02.2021.
//
#include "../IC_Angles.h"

void
IC_Angles(const cv::Mat &image, vx_keypoint_t *kp_buf, vx_size kp_size, vx_size kp_stride, vx_int32 *u_max_buf,
          vx_size u_max_size, vx_size u_max_stride)
{
    int m_01, m_10;
    const uchar *center;
    int step;
    int v_sum;
    int d;
    int val_plus;
    int val_minus;
    int v;
    int u;
    vx_keypoint_t kp;

    for(vx_size kp_i = 0; kp_i < kp_size; kp_i++) {
        kp = vxArrayItem(vx_keypoint_t, kp_buf, kp_i, kp_stride);

        m_01 = 0, m_10 = 0;

        center = &image.at<uchar>(cvRound(kp.y), cvRound(kp.x));

        // Treat the center line differently, v=0
        for (u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
            m_10 += u * center[u];

        // Go line by line in the circuI853lar patch
        step = (int) image.step1();
        for (v = 1; v <= HALF_PATCH_SIZE; ++v) {
            // Proceed over the two lines
            v_sum = 0;
            d = vxArrayItem(vx_int32, u_max_buf, v, u_max_stride);
            for (u = -d; u <= d; ++u) {
                val_plus = center[u + v * step];
                val_minus = center[u - v * step];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }
        kp.orientation = atan_f32((float)m_01, (float)m_10);
    }
}
