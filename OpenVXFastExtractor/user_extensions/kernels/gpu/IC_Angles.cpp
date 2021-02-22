//
// Created by denis on 2/21/21.
//

#include "../IC_Angles.h"
#include <cmath>

extern "C" void loadUMax();

void loadUMax() {
    std::vector<int>u_max(HALF_PATCH_SIZE + 1);

    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
        u_max[v] = cvRound(sqrt(hp2 - v * v));

// Make sure we are symmetric
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (u_max[v0] == u_max[v0 + 1])
            ++v0;
        u_max[v] = v0;
        ++v0;
    }
    loadUMax(u_max.data(), u_max.size());
}