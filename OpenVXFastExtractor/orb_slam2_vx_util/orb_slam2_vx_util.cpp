//
// Created by denis on 2/17/21.
//

#include "orb_slam2_vx_util.hpp"
#include <stdio.h>
#include <iostream>

extern "C" {
    #define ERROR_CHECK_STATUS( status ) { \
            vx_status status_ = (status); \
            if(status_ != VX_SUCCESS) { \
                printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
                exit(1); \
            } \
        }
    static int localCeil( double value ) {
        int i = (int)value;
        return i + (i < value);
    }
    static int localFloor( double value ) {
        int i = (int)value;
        return i - (i > value);
    }
    static int localRound( int value ) {
        return value;
    }
}

vx_array createUMax(vx_context context) {
    vx_array u_max = vxCreateArray(context, VX_TYPE_INT32, HALF_PATCH_SIZE + 1);
    std::vector<int> u_max_data(HALF_PATCH_SIZE + 1);

    int v, v0, vmax = localFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = localCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;


    for (v = 0; v <= vmax; ++v)
        u_max_data[v] = localRound(sqrt(hp2 - v * v));

    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (u_max_data[v0] == u_max_data[v0 + 1])
            ++v0;
        u_max_data[v] = v0;
        ++v0;
    }
    ERROR_CHECK_STATUS(vxAddArrayItems(u_max, HALF_PATCH_SIZE + 1, u_max_data.data(), sizeof(vx_int32)));
    for (auto e: u_max_data) {
        std::cout << e << std::endl;
    }
    return u_max;
}