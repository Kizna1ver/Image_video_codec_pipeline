aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} DIR_SRCS)
message("json kit:${DIR_SRCS}")
add_library(json ${DIR_SRCS})
