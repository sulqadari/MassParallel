add_library(${L_VECTOR}
	vector.c
)

target_include_directories(${L_VECTOR}
	PRIVATE ${PROJECT_SOURCE_DIR}
)