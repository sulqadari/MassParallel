add_executable(${EXE_MAIN} main.c)

target_include_directories(${EXE_MAIN}
	PRIVATE ${PROJECT_SOURCE_DIR}
)

target_link_libraries(${EXE_MAIN} ${L_VECTOR})