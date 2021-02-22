//
// Created by denis on 2/21/21.
//

#ifndef OPENVXFASTEXTRACTOR_UTIL_H
#define OPENVXFASTEXTRACTOR_UTIL_H

#include <stdlib.h>
#include <opencv2/core.hpp>

constexpr float atan2_p1 = 0.9997878412794807f*(float)(180/CV_PI);
constexpr float atan2_p3 = -0.3258083974640975f*(float)(180/CV_PI);
constexpr float atan2_p5 = 0.1555786518463281f*(float)(180/CV_PI);
constexpr float atan2_p7 = -0.04432655554792128f*(float)(180/CV_PI);

extern "C" {
#ifdef __EMSCRIPTEN__
inline float atan_f32(float y, float x)
    {
        float a = atan2(y, x) * 180.0f / CV_PI;
        if (a < 0.0f)
            a += 360.0f;
        if (a >= 360.0f)
            a -= 360.0f;
        return a; // range [0; 360)
    }
#else
inline float atan_f32(float y, float x)
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

#endif //OPENVXFASTEXTRACTOR_UTIL_H
