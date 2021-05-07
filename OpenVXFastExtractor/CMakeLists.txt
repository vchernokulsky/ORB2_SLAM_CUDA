cmake_minimum_required(VERSION 2.8.3)
project(OpenVXFastExtractor)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

message(STATUS "architecture: " ${CMAKE_SYSTEM_PROCESSOR} )

#add_definitions (-DIC_ANGLES_GPU)

string(APPEND CMAKE_CXX_FLAGS " -fPIC")
string(APPEND CMAKE_C_FLAGS " -fPIC")

set(VISIONWORKS_DIR 1)
set(OpenCV_VERSION 4.1.0)

message(STATUS "cxx flags: " ${CMAKE_CXX_FLAGS})
message(STATUS "c flags: " ${CMAKE_C_FLAGS})
message(STATUS "cuda flags: " ${CMAKE_CUDA_FLAGS})
message(STATUS "OpenCV version: " ${OpenCV_VERSION})

find_package(VisionWorks REQUIRED)
find_package(OpenCV ${OpenCV_VERSION} REQUIRED)
find_package(CUDA REQUIRED 10.2)

file(GLOB SOURCES "*.cpp" ".c")
file(GLOB HEADERS "*.h" "*.hpp")

add_subdirectory(user_extensions)
add_subdirectory(opencv_util)

include_directories(
    ${VisionWorks_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    user_extensions
    opencv_util
    /usr/local/cuda-10.2/include
)

add_executable(${PROJECT_NAME} ${SOURCES})
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
