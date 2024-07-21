cmake_minimum_required(VERSION 3.25)
project(VecAdd VERSION 0.1)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Ofast")

set(L_VECTOR "Vector")
set(EXE_MAIN "VectorApp")

add_subdirectory(vector)
add_subdirectory(src)