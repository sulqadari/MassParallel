cmake_minimum_required(VERSION 3.25)
project(VecAdd VERSION 0.1 LANGUAGES C CUDA)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Ofast")
set(BUILD_SHARED_LIBS NO)

option(BUILD_CUDA_VERSION "Building CUDA version" YES)

if(BUILD_CUDA_VERSION)
	add_compile_definitions(
		PUBLIC USE_CUDA
	)
endif()

add_subdirectory(vector)
add_subdirectory(src)
