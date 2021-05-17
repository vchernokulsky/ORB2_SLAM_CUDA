cmake_minimum_required(VERSION 3.8)
project(ORB_SLAM2_CUDA)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    string(APPEND CMAKE_CXX_FLAGS " -std=c++14 -O3 -fuse-ld=gold")
    string(APPEND CMAKE_C_FLAGS " -std=c11 -O3 -fuse-ld=gold")
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        string(APPEND CMAKE_CXX_FLAGS " -m64")
        string(APPEND CMAKE_C_FLAGS " -m64")
        string(APPEND CMAKE_C_FLAGS " -ffast-math -funroll-loops -fvect-cost-model=unlimited -ftree-loop-vectorize")
        string(APPEND CMAKE_CUDA_FLAGS " -cudart static -use_fast_math -gencode arch=compute_61,code=sm_61 -O3")
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        string(APPEND CMAKE_CXX_FLAGS " -march=armv8-a+fp+simd+crc -mcpu=cortex-a57+crc -ffast-math -funroll-loops -fvect-cost-model=unlimited -ftree-loop-vectorize")
        string(APPEND CMAKE_C_FLAGS " -mfloat-abi=hard -march=armv8-a+fp+simd+crc -mcpu=cortex-a57+crc -ffast-math -funroll-loops -fvect-cost-model=unlimited -ftree-loop-vectorize")
        string(APPEND CMAKE_CUDA_FLAGS " -cudart static -use_fast_math -gencode arch=compute_53,code=sm_53 -dc -O3")
    endif()
endif()

add_definitions(-DCOMPILEDWITHC11)

set(VISIONWORKS_DIR 1)
set(OpenCV_VERSION 4.1.0)

message(STATUS "architecture: " ${CMAKE_SYSTEM_PROCESSOR} )
message(STATUS "cxx flags: " ${CMAKE_CXX_FLAGS})
message(STATUS "c flags: " ${CMAKE_C_FLAGS})
message(STATUS "cuda flags: " ${CMAKE_CUDA_FLAGS} )
message(STATUS "OpenCV version: " ${OpenCV_VERSION})

find_package(VisionWorks REQUIRED)
find_package(OpenCV ${OpenCV_VERSION} REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(CUDA REQUIRED 10.0)
find_package(Pangolin REQUIRED)

include_directories(/usr/local/cuda-10.0/include)

add_subdirectory(opencv_util)
add_subdirectory(user_extensions)
add_subdirectory(ORB_SLAM2)
add_subdirectory(Examples)
