cmake_minimum_required(VERSION 3.10)
project(OpenVXFastExtractor)

IF(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -m64 -O3 -fuse-ld=gold")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -m64 -O3 -fuse-ld=gold")
ENDIF(CMAKE_BUILD_TYPE STREQUAL "Release")

message(STATUS "cxx flags:" ${CMAKE_CXX_FLAGS})
message(STATUS "c flags:" ${CMAKE_C_FLAGS})

set(OPENVX_DIR "/home/denis/OpenVX-sample-impl")
set(OpenVX_INCLUDE_DIRS "/home/denis/OpenVX-sample-impl/install/Linux/x64/Release/include")
set(OpenVX_LIBS_DIRS "/home/denis/OpenVX-sample-impl/install/Linux/x64/Release/bin")

find_package(OpenCV 3.2.0 REQUIRED)

file(GLOB SOURCES "*.cpp" "*.c")
file(GLOB HEADERS "*.h" "*.hpp")

add_subdirectory(opencv_util)
add_subdirectory(user_nodes)

include_directories(${OpenVX_INCLUDE_DIRS})

link_directories(${OpenVX_LIBS_DIRS})

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME}
    opencv_videoio
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui
    openvx
    opencv_util
    user_nodes
)
