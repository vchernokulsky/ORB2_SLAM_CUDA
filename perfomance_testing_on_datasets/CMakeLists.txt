cmake_minimum_required(VERSION 3.8)
project(ORB_SLAM2_CUDA)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

add_definitions(-DCOMPILEDWITHC11)

set(VISIONWORKS_DIR 1)
set(OpenCV_VERSION 4.1.0)

message(STATUS "architecture: " ${CMAKE_SYSTEM_PROCESSOR} )
message(STATUS "cxx flags: " ${CMAKE_CXX_FLAGS})
message(STATUS "c flags: " ${CMAKE_C_FLAGS})
message(STATUS "OpenCV version: " ${OpenCV_VERSION})

find_package(VisionWorks REQUIRED)
find_package(OpenCV ${OpenCV_VERSION} REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(CUDA REQUIRED 10.2)

include_directories(/usr/local/cuda-10.2/include)

add_subdirectory(opencv_util)
add_subdirectory(user_extensions)
add_subdirectory(ORB_SLAM2)
add_subdirectory(Examples)