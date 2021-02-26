#cmake_minimum_required(VERSION 3.10)
#project(OpenVXFastExtractor)
#
#IF(CMAKE_BUILD_TYPE STREQUAL "Release")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -m64 -O3 -fuse-ld=gold")
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -m64 -O3 -fuse-ld=gold")
#ENDIF(CMAKE_BUILD_TYPE STREQUAL "Release")
#
#message(STATUS "cxx flags:" ${CMAKE_CXX_FLAGS})
#message(STATUS "c flags:" ${CMAKE_C_FLAGS})
#
#set(OPENVX_DIR "/home/denis/OpenVX-sample-impl")
#set(OpenVX_INCLUDE_DIRS "/home/denis/OpenVX-sample-impl/install/Linux/x64/Release/include")
#set(OpenVX_LIBS_DIRS "/home/denis/OpenVX-sample-impl/install/Linux/x64/Release/bin")
#
#find_package(OpenCV 3.2.0 REQUIRED)
#
#file(GLOB SOURCES "*.cpp" "*.c")
#file(GLOB HEADERS "*.h" "*.hpp")
#
#add_subdirectory(opencv_util)
#add_subdirectory(user_extensions)
#
#include_directories(${OpenVX_INCLUDE_DIRS})
#
#link_directories(${OpenVX_LIBS_DIRS})
#
#add_executable(${PROJECT_NAME} ${SOURCES})
#target_link_libraries(${PROJECT_NAME}
#    opencv_videoio
#    opencv_core
#    opencv_imgproc
#    opencv_imgcodecs
#    opencv_highgui
#    openvx
#    opencv_util
#    user_extensions
#)

cmake_minimum_required(VERSION 3.17)
project(OpenVXFastExtractor)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

message(STATUS "architecture: " ${CMAKE_SYSTEM_PROCESSOR} )

add_definitions (-DIC_ANGLES_GPU)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.0/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "g++")
string( APPEND CMAKE_CUDA_FLAGS "-Xcompiler -O3 -cudart static -gencode arch=compute_53,code=sm_53 --use_fast_math --std=c++11 -Xptxas -O3")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    string(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -fuse-ld=gold")
    string(APPEND CMAKE_C_FLAGS "-std=c11 -O3 -fuse-ld=gold")
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        string(APPEND CMAKE_CXX_FLAGS " -m64")
        string(APPEND CMAKE_C_FLAGS " -m64")
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        string(APPEND CMAKE_CXX_FLAGS " -march=armv8-a+fp+simd+crc -mcpu=cortex-a57+crc -ffast-math -funroll-loops -fvect-cost-model=unlimited -ftree-loop-vectorize")
        string(APPEND CMAKE_C_FLAGS " -mfloat-abi=hard -march=armv8-a+fp+simd+crc -mcpu=cortex-a57+crc -ffast-math -funroll-loops -fvect-cost-model=unlimited -ftree-loop-vectorize")
        set(CMAKE_CUDA_FLAGS "-Xcompiler -march=armv8-a+fp+simd+crc -O3 -cudart static -gencode arch=compute_53,code=sm_53 --use_fast_math --std=c++11 -Xptxas -O3 --use_fast_math")
    endif()
endif()

set(CUDA_SEPARABLE_COMPILATION ON)
set(VISIONWORKS_DIR 1)
set(OpenCV_VERSION 4.1.0)

message(STATUS "cxx flags: " ${CMAKE_CXX_FLAGS})
message(STATUS "c flags: " ${CMAKE_C_FLAGS})
message(STATUS "cuda flags: " ${CMAKE_CUDA_FLAGS})
message(STATUS "OpenCV version: " ${OpenCV_VERSION})

find_package(VisionWorks REQUIRED)
find_package(OpenCV ${OpenCV_VERSION} REQUIRED)
find_package(CUDA REQUIRED 10.0)

enable_language(CUDA)

set(CUDA_SEPARABLE_COMPILATION ON)

file(GLOB SOURCES "*.cpp" ".c")
file(GLOB HEADERS "*.h" "*.hpp")

add_subdirectory(user_extensions)
add_subdirectory(opencv_util)

cuda_include_directories(
    ${VisionWorks_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    user_extensions
    opencv_util
)

cuda_add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME}
    opencv_videoio
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui
    ${VisionWorks_LIBRARIES}
    opencv_util
    user_extensions
)
