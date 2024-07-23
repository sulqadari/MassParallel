if (BUILD_CUDA_VERSION)
	add_executable(VectorApp main.cu)
else()
	add_executable(VectorApp main.c)
endif()


target_include_directories(VectorApp
	PRIVATE ${PROJECT_SOURCE_DIR}
)

target_link_libraries(VectorApp Vector)