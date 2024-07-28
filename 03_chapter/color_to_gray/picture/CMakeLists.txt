if (BUILD_CUDA_VERSION)
	add_library(bmpImage bmp_image.c)
else()
	add_library(bmpImage bmp_image.c)
endif()

target_include_directories(bmpImage
	PRIVATE ${PROJECT_SOURCE_DIR}
)