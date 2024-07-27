set(EXECUTABLE_NAME		"ColorToGray")


if (BUILD_CUDA_VERSION)
	set(${EXECUTABLE_NAME}		"gpu_ColorToGray")
	add_executable(${EXECUTABLE_NAME} main.cu)
else()
	add_executable(${EXECUTABLE_NAME} main.c)
endif()


target_include_directories(${EXECUTABLE_NAME}
	PRIVATE ${PROJECT_SOURCE_DIR}
)