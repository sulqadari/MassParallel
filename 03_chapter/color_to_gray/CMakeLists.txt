cmake_minimum_required(VERSION 3.25)
project(ColorToGray VERSION 0.1 LANGUAGES C CUDA)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O0 -g3")
option(BUILD_SHARED_LIBS "Build libraries as shared" NO)

option(BUILD_CUDA_VERSION "Building CUDA version" NO)

set(EXECUTABLE_NAME "ColorToGrayscale")

add_compile_definitions(EXECUTABLE_NAME="${EXECUTABLE_NAME}")

if (BUILD_CUDA_VERSION)
	set(${EXECUTABLE_NAME} "ColorToGrayscale_gpu")
endif()

add_subdirectory(picture)
add_subdirectory(src)