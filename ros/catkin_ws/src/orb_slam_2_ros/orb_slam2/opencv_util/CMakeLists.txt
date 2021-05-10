get_filename_component(project_dir ${CMAKE_CURRENT_LIST_DIR} NAME)
project(${project_dir})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

file(GLOB SOURCES "*.cpp" "*.c")
file(GLOB HEADERS "*.h" "*.hpp")

include_directories(
    ${VisionWorks_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${project_dir}
)

add_library(${PROJECT_NAME} STATIC ${SOURCES})
target_link_libraries(${PROJECT_NAME}
    opencv_core
    ${VisionWorks_LIBRARIES}
)