get_filename_component(project_dir ${CMAKE_CURRENT_LIST_DIR} NAME)
project(${project_dir})

set(OPENVX_DIR "/home/denis/OpenVX-sample-impl")
set(OpenVX_INCLUDE_DIRS "/home/denis/OpenVX-sample-impl/install/Linux/x64/Release/include")
set(OpenVX_LIBS_DIRS "/home/denis/OpenVX-sample-impl/install/Linux/x64/Release/bin")

find_package(OpenCV 3.2.0 REQUIRED)

file(GLOB SOURCES "*/*/*.cpp" "*/*/*.c" "*/*.cpp" "*/*.c" "*.cpp" "*.c")
file(GLOB HEADERS "*/*/*.h" "*/*/*.hpp" "*/*.h" "*/*.hpp" "*.h" "*.hpp")

include_directories(
        ${OpenVX_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${project_dir}
)

add_library(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME}
    opencv_core
    openvx
)