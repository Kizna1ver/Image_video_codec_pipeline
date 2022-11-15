# GPUJPEG CMake
# option(DISABLE_CPP "Disable C++ code (currently image delegates)" OFF)
# option(HUFFMAN_GPU_CONST_TABLES "Enable Huffman GPU decoder quick const tables" OFF)
# file(GLOB_RECURSE GPUJPEG_HEADER_FILES ${PIPELINE_3RDPARTY_DIR}/gpujpeg/include/*.h)
# file(GLOB_RECURSE GPUJPEG_SRC_FILES ${PIPELINE_3RDPARTY_DIR}/gpujpeg/src/*.c 
#  ${PIPELINE_3RDPARTY_DIR}/gpujpeg/src/*.cu 
#  ${PIPELINE_3RDPARTY_DIR}/gpujpeg/src/*.cpp
#  )
# if(NOT DISABLE_CPP)
#     file(GLOB_RECURSE CPP_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
# endif()
# if(HUFFMAN_GPU_CONST_TABLES)
#     message("HUFFMAN_GPU_CONST_TABLES")
#     add_definitions("-DHUFFMAN_GPU_CONST_TABLES")
#     set(COMPILED_OPTIONS "${COMPILED_OPTIONS Huffman_const_tables")
# endif()

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src DIR_SRCS)
# find_package(CUDA)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})
add_library(gpujpeg ${DIR_SRCS})
set_property(TARGET gpujpeg PROPERTY CUDA_ARCHITECTURES OFF)
