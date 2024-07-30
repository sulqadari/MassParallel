if (BUILD_CUDA_VERSION)
	add_library(bmpImage bmp_image.cu)
else()
	add_library(bmpImage bmp_image.c)
endif()

target_include_directories(bmpImage
	PRIVATE ${PROJECT_SOURCE_DIR}
)

# set_target_properties( bmpImage
#                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
#                       )