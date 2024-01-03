cmake_minimum_required(VERSION 3.25)
project(VectorAddition LANGUAGES C CUDA VERSION 1.0)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall			\
	-Wextra -Wno-unused-parameter					\
	-Wno-missing-field-initializers -Wmissing-declarations	\
	-Wredundant-decls -Wcast-align							\
	-Wformat -Wformat-security"
)

add_subdirectory(cpu)
add_subdirectory(gpu)