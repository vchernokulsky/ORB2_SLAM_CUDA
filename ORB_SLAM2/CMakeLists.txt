cmake_minimum_required(VERSION 3.8)
project(ORB_SLAM2)

set(CMAKE_CXX_STANDARD 11)

LIST(APPEND CMAKE_MODULE_PATH ../../cmake_modules)

find_package(OpenCV 4.1.0 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)

file(GLOB SOURCES 
	"src/*.cpp" 
	"src/*.cc"
	"src/*.c"
)
file(GLOB HEADERS 
	"inc/*.h" 
	"inc/*.hpp"
)

include_directories(
	${PROJECT_SOURCE_DIR}/..
	${PROJECT_SOURCE_DIR}/inc
	${EIGEN3_INCLUDE_DIR}
	${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../libs)
add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_link_libraries(${PROJECT_NAME} PUBLIC
	${OpenCV_LIBS}
	${EIGEN3_LIBS}
	${PROJECT_SOURCE_DIR}/../Thirdparty/DBoW2/lib/libDBoW2.so
	${PROJECT_SOURCE_DIR}/../Thirdparty/g2o/lib/libg2o.so
)

