add_executable(${CMAKE_PROJECT_NAME}GPU main.cu)

target_compile_features(${CMAKE_PROJECT_NAME}GPU PUBLIC c_std_11)

# The instructions below reduce performance by two.

# set_target_properties(${CMAKE_PROJECT_NAME}GPU
# 	PROPERTIES CUDA_SEPARABLE_COMPILATION ON
# )

# set_property(TARGET ${CMAKE_PROJECT_NAME}GPU PROPERTY
# 	CUDA_SEPARABLE_COMPILATION ON
# )