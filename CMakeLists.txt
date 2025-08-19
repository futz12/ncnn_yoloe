cmake_minimum_required(VERSION 3.31)
project(ncnn_yoloe)

set(CMAKE_CXX_STANDARD 20)

set(ncnn_DIR 3rdparty/ncnn/lib/cmake/ncnn)
set(ncnn_INCLUDE_DIRS 3rdparty/ncnn/include)
find_package(ncnn REQUIRED)
include_directories(${ncnn_INCLUDE_DIRS})

set(OpenCV_DIR 3rdparty/ocvm/x64/vc17/staticlib)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ncnn_yoloe main.cpp MobileClip.cpp SimpleTokenizer.cpp YOLOE.cpp)

target_link_libraries(ncnn_yoloe ncnn ${OpenCV_LIBS})
