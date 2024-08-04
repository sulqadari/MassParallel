cmake_minimum_required(VERSION 3.25)
project(03_MatrixMultiplication VERSION 0.1 LANGUAGES C CUDA)


option(BUILD_SHARED_LIBS "Build libraries as shared" NO)
option(BUILD_CUDA_VERSION "BUilding CUDA version" NO)

if (BUILD_CUDA_VERSION)
	set(EXECUTABLE_NAME "MatrixMul_gpu")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress 2464")
	set(sources ${PROJECT_SOURCE_DIR}/src/main.cu)
else()
	set(EXECUTABLE_NAME "MatrixMul")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O0 -g3 -fsanitize=address -static-libasan -fno-omit-frame-pointer")
	set(sources ${PROJECT_SOURCE_DIR}/src/main.c)
endif()

add_executable(${EXECUTABLE_NAME}
	${sources}
	${PROJECT_SOURCE_DIR}/src/common.h
)