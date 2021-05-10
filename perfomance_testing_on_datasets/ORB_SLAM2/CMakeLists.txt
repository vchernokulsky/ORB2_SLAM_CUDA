get_filename_component(project_dir ${CMAKE_CURRENT_LIST_DIR} NAME)
project(${project_dir})

list(APPEND CMAKE_MODULE_PATH ../cmake_modules)

file(GLOB SOURCES
	"src/*.cpp" 
	"src/*.cc"
	"src/*.c"
)
file(GLOB HEADERS 
	"inc/*.h" 
	"inc/*.hpp"
)

find_package(Pangolin REQUIRED)

include_directories(
	${PROJECT_SOURCE_DIR}/..
	${PROJECT_SOURCE_DIR}/inc
	${EIGEN3_INCLUDE_DIR}
	${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} STATIC ${SOURCES})

target_link_libraries(${PROJECT_NAME}
	${OpenCV_LIBS}
	${EIGEN3_LIBS}
	${PROJECT_SOURCE_DIR}/../Thirdparty/DBoW2/lib/libDBoW2.so
	${PROJECT_SOURCE_DIR}/../Thirdparty/g2o/lib/libg2o.so
	user_extensions
	opencv_util
)

