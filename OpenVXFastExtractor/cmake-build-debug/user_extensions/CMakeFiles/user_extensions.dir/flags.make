# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# compile CUDA with /usr/local/cuda-10.0/bin/nvcc
# compile CXX with /usr/bin/c++
CUDA_FLAGS = -Xcompiler -O3 -cudart static -gencode arch=compute_53,code=sm_53 --use_fast_math --std=c++11 -Xptxas -O3 -g   -std=c++14

CUDA_DEFINES = -DIC_ANGLES_GPU

CUDA_INCLUDES = -I/home/denis/ORB2_SLAM_CUDA/OpenVXFastExtractor/cmake-build-debug/user_extensions -I/home/denis/ORB2_SLAM_CUDA/OpenVXFastExtractor/user_extensions -I/usr/local/cuda/include -isystem=/usr/local/include/opencv4 

CXX_FLAGS = -g  

CXX_DEFINES = -DIC_ANGLES_GPU

CXX_INCLUDES = -I/home/denis/ORB2_SLAM_CUDA/OpenVXFastExtractor/cmake-build-debug/user_extensions -I/home/denis/ORB2_SLAM_CUDA/OpenVXFastExtractor/user_extensions -I/usr/local/cuda/include -isystem /usr/local/include/opencv4 

