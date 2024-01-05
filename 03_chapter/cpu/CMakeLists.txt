add_executable(${CMAKE_PROJECT_NAME}CPU main.c)
target_compile_features(${CMAKE_PROJECT_NAME}CPU PUBLIC c_std_11)