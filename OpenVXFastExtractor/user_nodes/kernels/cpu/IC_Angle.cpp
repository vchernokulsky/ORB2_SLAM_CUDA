//
// Created by denis on 16.02.2021.
//

#include "IC_Angle.hpp"

static const float atan2_p1 = 0.9997878412794807f*(float)(180/CV_PI);
static const float atan2_p3 = -0.3258083974640975f*(float)(180/CV_PI);
static const float atan2_p5 = 0.1555786518463281f*(float)(180/CV_PI);
static const float atan2_p7 = -0.04432655554792128f*(float)(180/CV_PI);

extern "C" {
    #ifdef __EMSCRIPTEN__
    static float atan_f32(float y, float x)
    {
        float a = atan2(y, x) * 180.0f / CV_PI;
        if (a < 0.0f)
            a += 360.0f;
        if (a >= 360.0f)
            a -= 360.0f;
        return a; // range [0; 360)
    }
    #else
    static float atan_f32(float y, float x)
    {
        float ax = abs(x), ay = abs(y);
        float a, c, c2;
        if( ax >= ay )
        {
            c = ay/(ax + (float)__DBL_EPSILON__);
            c2 = c*c;
            a = (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
        }
        else
        {
            c = ax/(ay + (float)__DBL_EPSILON__);
            c2 = c*c;
            a = 90.f - (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
        }
        if( x < 0 )
            a = 180.f - a;
        if( y < 0 )
            a = 360.f - a;
        return a;
    }
    #endif
}

void IC_Angles(const cv::Mat& image, vx_keypoint_t *kp_buf, vx_size kp_size, vx_size kp_stride, const vx_int32 *u_max_buf)
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
    vx_keypoint_t * kp;

    for(vx_size kp_i = 0; kp_i < kp_size; kp_i++) {
        kp = (vx_keypoint_t *) (kp_buf + kp_i * kp_stride );

        m_01 = 0, m_10 = 0;

        center = &image.at<uchar>(cvRound(kp->y), cvRound(kp->x));

        // Treat the center line differently, v=0
        for (u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
            m_10 += u * center[u];

        // Go line by line in the circuI853lar patch
        step = (int) image.step1();
        for (v = 1; v <= HALF_PATCH_SIZE; ++v) {
            // Proceed over the two lines
            v_sum = 0;
            d = u_max_buf[v];
            for (u = -d; u <= d; ++u) {
                val_plus = center[u + v * step];
                val_minus = center[u - v * step];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }
        kp->orientation = atan_f32((float)m_01, (float)m_10);
    }
}
