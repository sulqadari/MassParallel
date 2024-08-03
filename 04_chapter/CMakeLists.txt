cmake_minimum_required(VERSION 3.25)
project(QueryDeviceInfo VERSION 0.1 LANGUAGES C CUDA)

option(BUILD_SHARED_LIBS "Build libraries as shared" NO)
set(EXECUTABLE_NAME "DeviceInfo")

#suppresses annoying 'deprecated conversation from string to char' warning.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress 2464")

add_executable(${EXECUTABLE_NAME} ${PROJECT_SOURCE_DIR}/src/main.cu)