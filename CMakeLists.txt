cmake_minimum_required(VERSION 3.8)
project(ORB_SLAM2_CUDA)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

add_definitions(-DCOMPILEDWITHC11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

add_subdirectory(ORB_SLAM2)
add_subdirectory(Examples)
