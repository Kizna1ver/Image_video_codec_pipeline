cmake_minimum_required(VERSION 3.8.0 FATAL_ERROR)
SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(image_video_codec_pipeline LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release")
  message("default build type: ${CMAKE_BUILD_TYPE}")
else()
  message("use specified build type: ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_CXX_FLAGS_DEBUG
    "$ENV{CXXFLAGS} -g -Wall -fno-omit-frame-pointer -fsanitize-recover=address"
)
find_package(CUDA REQUIRED)
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -g -W
# find nvcodec lib
message("CUDA_CUDA_LIBRARY: ${CUDA_CUDA_LIBRARY}")
message("CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
message("CUDA_TOOLKIT_INCLUDE_DIR: ${CUDA_INCLUDE_DIRS}") # This variable will be empty if comment find_package(CUDA REQUIRED)

# find_library(CUVID_LIB nvcuvid) find_library(NVENCODEAPI_LIB nvidia-encode)
# message("${CUVID_LIB}") message("${NVENCODEAPI_LIB}")
set(PIPELINE_3RDPARTY_DIR "/home/ljy/jianyu.liu/project/image-video-codec-pipeline-feature-test/3rdparty" CACHE INTERNAL "This is description") 
add_subdirectory(3rdparty/gpujpeg)
add_subdirectory(3rdparty/nvcodec)
add_subdirectory(3rdparty/json)

# find ffmpeg cedec lib
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(PC_AVCODEC REQUIRED IMPORTED_TARGET libavcodec)
  pkg_check_modules(PC_AVFORMAT REQUIRED IMPORTED_TARGET libavformat)
  pkg_check_modules(PC_AVUTIL REQUIRED IMPORTED_TARGET libavutil)
  pkg_check_modules(PC_SWRESAMPLE REQUIRED IMPORTED_TARGET libswresample)

  set(NV_FFMPEG_HDRS ${PC_AVCODEC_INCLUDE_DIRS})
  find_library(
    AVCODEC_LIBRARY
    NAMES avcodec
    HINTS ${PC_AVCODEC_LIBDIR} ${PC_AVCODEC_LIBRARY_DIRS})
  find_library(
    AVFORMAT_LIBRARY
    NAMES avformat
    HINTS ${PC_AVFORMAT_LIBDIR} ${PC_AVFORMAT_LIBRARY_DIRS})
  find_library(
    AVUTIL_LIBRARY
    NAMES avutil
    HINTS ${PC_AVUTIL_LIBDIR} ${PC_AVUTIL_LIBRARY_DIRS})
  find_library(
    SWRESAMPLE_LIBRARY
    NAMES swresample
    HINTS ${PC_SWRESAMPLE_LIBDIR} ${PC_SWRESAMPLE_LIBRARY_DIRS})
  set(AVCODEC_LIB ${AVCODEC_LIBRARY})
  set(AVFORMAT_LIB ${AVFORMAT_LIBRARY})
  set(AVUTIL_LIB ${AVUTIL_LIBRARY})
  set(SWRESAMPLE_LIB ${SWRESAMPLE_LIBRARY})
endif()

# find dataset2
# find_package(ament_cmake REQUIRED)
# find_package(dataset2 REQUIRED)
# find_package(pandar_msgs REQUIRED)
# find_package(sensor_msgs REQUIRED)

file(GLOB_RECURSE PIPELINE_HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB_RECURSE PIPELINE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
message("src file:${PIPELINE_SRC_FILES}")

add_executable(${CMAKE_PROJECT_NAME} ${PIPELINE_HEADER_FILES}
                                     ${PIPELINE_SRC_FILES})
target_include_directories(
  ${CMAKE_PROJECT_NAME}
  PUBLIC ${CUDA_INCLUDE_DIRS}
         ${PIPELINE_3RDPARTY_DIR}/gpujpeg/include
         ${PIPELINE_3RDPARTY_DIR}/nvcodec/include
         ${PIPELINE_3RDPARTY_DIR}/nvcodec_lib/include
         ${PIPELINE_3RDPARTY_DIR}/json/include
         ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(
  ${CMAKE_PROJECT_NAME}
  nvcodec
  gpujpeg
  # dataset2
  json
  ${CUDA_CUDA_LIBRARY}
  ${AVCODEC_LIB}
  ${AVFORMAT_LIB}
  ${AVUTIL_LIB}
  ${SWRESAMPLE_LIB}
  :libnvcuvid.so.1
  :libnvidia-encode.so.1
  ) # library

# ament_target_dependencies(${CMAKE_PROJECT_NAME} sensor_msgs pandar_msgs)
