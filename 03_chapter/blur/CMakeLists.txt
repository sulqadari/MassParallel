cmake_minimum_required(VERSION 3.25)
project(ColorToGray 0.1 LANGUAGES C CUDA)

set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} -Wall -Ofast)

option(BUILD_SHARED_LIBS "Build libraries as shared" OFF)
option(BUILD_CUDA_VERSION "Building CUDA version" OFF)

add_subdirectory(src)