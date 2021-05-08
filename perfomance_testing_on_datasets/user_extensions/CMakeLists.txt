get_filename_component(project_dir ${CMAKE_CURRENT_LIST_DIR} NAME)
project(${project_dir})

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.0/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "g++")

file(GLOB_RECURSE CPP_SOURCES *.cpp)
file(GLOB_RECURSE CUDA_SOURCES "*.cu")

find_package(CUDA REQUIRED 10.0)

enable_language(CUDA)

cuda_include_directories(
    ${project_dir}
    ${VisionWorks_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/cuda-10.0/include
)

cuda_add_library(${PROJECT_NAME} STATIC ${CUDA_SOURCES} ${CPP_SOURCES}
)
target_link_libraries(${PROJECT_NAME}
    opencv_core
    ${VisionWorks_LIBRARIES}
    nvToolsExt
)

