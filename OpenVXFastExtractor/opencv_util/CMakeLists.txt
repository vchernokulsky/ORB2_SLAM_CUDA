#get_filename_component(project_dir ${CMAKE_CURRENT_LIST_DIR} NAME)
#project(${project_dir})
#
#set(OPENVX_DIR "/home/denis/OpenVX-sample-impl")
#set(OpenVX_INCLUDE_DIRS "/home/denis/OpenVX-sample-impl/install/Linux/x64/Release/include")
#set(OpenVX_LIBS_DIRS "/home/denis/OpenVX-sample-impl/install/Linux/x64/Release/bin")
#
#find_package(OpenCV ${OpenCV_VERSION} REQUIRED)
#
#file(GLOB SOURCES "*.cpp" "*.c")
#file(GLOB HEADERS "*.h" "*.hpp")
#
#include_directories(${OpenVX_INCLUDE_DIRS})
#include_directories(${OpenCV_INCLUDE_DIRS})
#
#add_library(${PROJECT_NAME} ${SOURCES})
#target_link_libraries(${PROJECT_NAME}
#    opencv_core
#    openvx
#)

get_filename_component(project_dir ${CMAKE_CURRENT_LIST_DIR} NAME)
project(${project_dir})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

file(GLOB SOURCES "*.cpp" "*.c")
file(GLOB HEADERS "*.h" "*.hpp")

cuda_include_directories(
    ${VisionWorks_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${project_dir}
)

cuda_add_library(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME}
    opencv_core
    ${VisionWorks_LIBRARIES}
)