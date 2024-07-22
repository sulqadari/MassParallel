if (BUILD_CUDA_VERSION)
	add_library(Vector vector.cu)
else()
	add_library(Vector vector.c)
endif()

target_include_directories(Vector
	PRIVATE ${PROJECT_SOURCE_DIR}
)