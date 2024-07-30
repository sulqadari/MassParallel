if (BUILD_CUDA_VERSION)
	add_executable(${EXECUTABLE_NAME} main.cu)
else()
	add_executable(${EXECUTABLE_NAME} main.c)
endif()

target_include_directories(${EXECUTABLE_NAME}
	PRIVATE ${PROJECT_SOURCE_DIR}
)

target_link_libraries(${EXECUTABLE_NAME} bmpImage)