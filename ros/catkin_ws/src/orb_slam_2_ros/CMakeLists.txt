cmake_minimum_required(VERSION 2.8.3)
project(orb_slam2_ros)

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
  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    string(APPEND CMAKE_CXX_FLAGS " -march=armv8-a+fp+simd+crc -mcpu=cortex-a57+crc -ffast-math -funroll-loops -fvect-cost-model=unlimited -ftree-loop-vectorize")
    string(APPEND CMAKE_C_FLAGS " -mfloat-abi=hard -march=armv8-a+fp+simd+crc -mcpu=cortex-a57+crc -ffast-math -funroll-loops -fvect-cost-model=unlimited -ftree-loop-vectorize")
    string(APPEND CMAKE_CUDA_FLAGS " -cudart static -use_fast_math -gencode arch=compute_53,code=sm_53 -dc -O3")
  endif()
endif()

string(APPEND CMAKE_CXX_FLAGS " -fPIC")
string(APPEND CMAKE_C_FLAGS " -fPIC")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/orb_slam2/cmake_modules)

set(VISIONWORKS_DIR 1)

find_package(VisionWorks REQUIRED)
find_package(OpenCV 4.1.0 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(CUDA REQUIRED 10.2)

enable_language(CUDA)

add_subdirectory(${PROJECT_SOURCE_DIR}/orb_slam2/opencv_util)
add_subdirectory(${PROJECT_SOURCE_DIR}/orb_slam2/user_extensions)

find_package (catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        cv_bridge
        image_transport
        tf
        sensor_msgs
        dynamic_reconfigure
        message_generation
        )

include_directories(/usr/local/cuda-10.2/include)

set (DYNAMIC_RECONFIGURE_PATH ros/config/dynamic_reconfigure.cfg)
execute_process(COMMAND chmod a+x ${DYNAMIC_RECONFIGURE_PATH}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE cmd_result
        OUTPUT_VARIABLE cmd_ver)
message(STATUS "Chmod a+x the dynamic_reconfigure file")

generate_dynamic_reconfigure_options(
        ${DYNAMIC_RECONFIGURE_PATH}
)

set(LIBS_ORBSLAM
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${PROJECT_SOURCE_DIR}/orb_slam2/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/orb_slam2/Thirdparty/g2o/lib/libg2o.so
        opencv_util
        user_extensions
        )

set(LIBS_ROS
        ${PROJECT_SOURCE_DIR}/orb_slam2/lib/lib${PROJECT_NAME}.so
        ${OpenCV_LIBS}
        ${catkin_LIBRARIES}
        )

add_service_files(
        FILES
        SaveMap.srv
)

generate_messages(
        DEPENDENCIES
        std_msgs
)

catkin_package (
        CATKIN_DEPENDS roscpp rospy std_msgs cv_bridge image_transport tf sensor_msgs dynamic_reconfigure message_runtime
        LIBRARIES {PROJECT_NAME} libDBoW2 libg2o
)

include_directories(
        ${PROJECT_SOURCE_DIR}/orb_slam2
        ${PROJECT_SOURCE_DIR}/orb_slam2/ORB_SLAM2
        ${PROJECT_SOURCE_DIR}/orb_slam2/ORB_SLAM2/inc
        ${PROJECT_SOURCE_DIR}/ros/include
        ${EIGEN3_INCLUDE_DIR}
        ${catkin_INCLUDE_DIRS}
)

include(${PROJECT_SOURCE_DIR}/orb_slam2/Thirdparty/DBoW2/CMakeLists.txt)
include(${PROJECT_SOURCE_DIR}/orb_slam2/Thirdparty/g2o/CMakeLists.txt)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/orb_slam2/lib)

file(GLOB ORB_SLAM2_SOURCES
        "${PROJECT_SOURCE_DIR}/orb_slam2/ORB_SLAM2/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/orb_slam2/ORB_SLAM2/src/*.cc"
        "${PROJECT_SOURCE_DIR}/orb_slam2/ORB_SLAM2/src/*.c"
        )

add_library(${PROJECT_NAME} SHARED
        ${ORB_SLAM2_SOURCES}
        )

add_dependencies (${PROJECT_NAME} g2o DBoW2)

target_link_libraries(${PROJECT_NAME}
        ${LIBS_ORBSLAM}
        )

# map serialization addition - library boost serialization
message(STATUS "Compile With map save/load function")
find_library(BOOST_SERIALIZATION boost_serialization)
if (NOT BOOST_SERIALIZATION)
  message(FATAL_ERROR "Can't find libboost_serialization")
endif()
target_link_libraries(${PROJECT_NAME} ${BOOST_SERIALIZATION})

add_executable (${PROJECT_NAME}_mono
        ros/src/MonoNode.cc
        ros/src/Node.cc
        )
add_dependencies (${PROJECT_NAME}_mono ${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

target_link_libraries(${PROJECT_NAME}_mono
        ${LIBS_ROS}
        )

add_executable (${PROJECT_NAME}_stereo
        ros/src/StereoNode.cc
        ros/src/Node.cc
        )
add_dependencies (${PROJECT_NAME}_stereo ${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

target_link_libraries(${PROJECT_NAME}_stereo
        ${LIBS_ROS}
        )

add_executable (${PROJECT_NAME}_rgbd
        ros/src/RGBDNode.cc
        ros/src/Node.cc
        )
add_dependencies (${PROJECT_NAME}_rgbd ${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

target_link_libraries(${PROJECT_NAME}_rgbd
        ${LIBS_ROS}
        )

install(TARGETS ${PROJECT_NAME}_mono ${PROJECT_NAME}_stereo ${PROJECT_NAME}_rgbd
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

install(TARGETS ${PROJECT_NAME} DBoW2 g2o
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

install(DIRECTORY ros/launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/ros/launch
        )

install(DIRECTORY ros/config/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/ros/config
        )

install(DIRECTORY orb_slam2/config/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/orb_slam2/config
        )

install(DIRECTORY orb_slam2/Vocabulary/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/orb_slam2/Vocabulary
        )
