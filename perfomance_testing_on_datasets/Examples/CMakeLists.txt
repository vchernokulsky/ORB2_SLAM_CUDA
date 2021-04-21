cmake_minimum_required(VERSION 3.8)
project(Examples)

find_package(Pangolin REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}/..
        ${PROJECT_SOURCE_DIR}/../ORB_SLAM2/inc
	${Pangolin_INCLUDE_DIRS}
)
link_libraries(
	ORB_SLAM2
	${Pangolin_LIBRARIES}
)

add_executable(rgbd_tum RGB-D/rgbd_tum.cc)
add_executable(stereo_kitti Stereo/stereo_kitti.cc)
add_executable(stereo_euroc Stereo/stereo_euroc.cc)
add_executable(mono_tum Monocular/mono_tum.cc)
add_executable(mono_kitti Monocular/mono_kitti.cc)
add_executable(mono_euroc Monocular/mono_euroc.cc)
