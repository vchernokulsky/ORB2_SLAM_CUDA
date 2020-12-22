cmake_minimum_required(VERSION 3.8)
project(Examples)

LIST(APPEND CMAKE_MODULE_PATH ../../cmake_modules)

set(CMAKE_CXX_STANDARD 11)
set("LIB_ORB_SLAM2" ${PROJECT_SOURCE_DIR}/../libs/libORB_SLAM2.so)

add_definitions(-DCOMPILEDWITHC11)

find_package(Eigen3 3.1.0 REQUIRED)
find_package(OpenCV 4.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
	${PROJECT_SOURCE_DIR}/..
	${PROJECT_SOURCE_DIR}/../ORB_SLAM2/inc
	${EIGEN3_INCLUDE_DIR}
	${OpenCV_INCLUDE_DIRS}
	${Pangolin_INCLUDE_DIRS}
)

set("LIBS" 
	${LIB_ORB_SLAM2}
	${EIGEN3_LIBS}
	${OpenCV_LIBS}
	${Pangolin_LIBRARIES}
	${PROJECT_SOURCE_DIR}/../Thirdparty/DBoW2/lib/libDBoW2.so
	${PROJECT_SOURCE_DIR}/../Thirdparty/g2o/lib/libg2o.so
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../bin)

add_executable(rgbd_tum RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${LIBS})

add_executable(stereo_kitti Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${LIBS})

add_executable(stereo_euroc Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${LIBS})

add_executable(mono_tum Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${LIBS})

add_executable(mono_kitti Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${LIBS})

add_executable(mono_euroc Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${LIBS})
