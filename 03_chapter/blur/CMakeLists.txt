cmake_minimum_required(VERSION 3.25)
project(ColorToGray VERSION 0.1 LANGUAGES C CUDA)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O0 -g3")
option(BUILD_SHARED_LIBS "Build libraries as shared" NO)

option(BUILD_CUDA_VERSION "Building CUDA version" YES)

if(BUILD_CUDA_VERSION)
	set(EXECUTABLE_NAME "Blurring_gpu")
	add_compile_definitions(EXECUTABLE_NAME="${EXECUTABLE_NAME}" USE_CUDA)

	#suppresses annoying 'deprecated conversation from string to char' warning.
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress 2464")
else()
	set(EXECUTABLE_NAME "Blurring")
	add_compile_definitions(EXECUTABLE_NAME="${EXECUTABLE_NAME}")
endif()

add_subdirectory(picture)
add_subdirectory(src)