cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(PROJECT_NAME ai_doberbot)
project(${PROJECT_NAME})
enable_language(CUDA)


set(CMAKE_CXX_COMPILIER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -Wno-write-strings")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

#cuda
#export PATH=/usr/local/cuda-11.0/bin:$PATH
#include_directories(/usr/local/cuda/include)
#link_directories(/usr/local/cuda/lib64)
find_package(CUDA REQUIRED)
message("-- CUDA version: ${CUDA_VERSION}")

set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
    -O3
        -gencode arch=compute_53,code=sm_53
        -gencode arch=compute_62,code=sm_62
)
if(CUDA_VERSION_MAJOR GREATER 9)
        message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_72")

        set(
                CUDA_NVCC_FLAGS
                ${CUDA_NVCC_FLAGS};
                -gencode arch=compute_72,code=sm_72
        )

endif()

# tensorRT
message("CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} /usr/include
  PATH_SUFFIXES include)
find_path(TENSORRT_INCLUDE_DIR NvInferPlugin.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64 lib/aarch64-linux-gnu)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64 lib/aarch64-linux-gnu)
  find_library(TENSORRT_LIBRARY_PARSER nvparsers
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64 lib/aarch64-linux-gnu)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_PARSER})
message(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR
    "Cannot find TensorRT library.")
endif()

# librealsense2
find_package(realsense2 REQUIRED)
# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# LIBREALSEANSE LINKING

#opencv
#set(OpenCV_DIR /opt/opencv-4.1.2/share/OpenCV)
find_package(OpenCV REQUIRED)



# l2norm_helper plugin
add_subdirectory(trt_l2norm_helper_facenet)
include_directories(
  trt_l2norm_helper_facenet
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${TENSORRT_INCLUDE_DIR}
)

#detector lib (cuda)
file(GLOB_RECURSE sources modules_facenet/*.h modules_facenet/*.cpp modules_yolo/*.hpp modules_yolo/*.cpp modules_yolo/*.h modules_yolo/*.cu utils/*.h utils/*.hpp utils/*.cpp)
add_library(detector SHARED ${sources})
target_include_directories(detector PRIVATE modules_facenet/ utils/ modules_yolo/ ${OpenCV_INCLUDE_DIRS} ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_link_libraries(detector nvinfer nvinfer_plugin nvparsers "stdc++fs")

#detector
add_executable(${PROJECT_NAME} detector/doberBot.cpp)				  
target_include_directories(${PROJECT_NAME} PRIVATE modules_facenet/ utils/ modules_yolo/ ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_link_libraries(${PROJECT_NAME} detector ${OpenCV_LIBS} ${TENSORRT_LIBRARY} trt_l2norm_helper )


# REALSENSE
#include_directories(include ${realsense_INCLUDE_DIR})
#target_include_directories(${PROJECT_NAME} PRIVATE ${realsense_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${realsense2_LIBRARY})
