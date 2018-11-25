cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(raytracer LANGUAGES CXX CUDA)

add_subdirectory(ext/Progression)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/ext/Progression/cmake")
include(Progression)
message("${PROGRESSION_INCLUDE_DIRS}")

set(
    SRCS
    src/main.cu
    #src/common.h
    #src/raytracer.h
    #src/raytracer.cu
    #src/shapes.h
)
 
add_executable(${PROJECT_NAME} ${SRCS})

include_directories(src/)

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
target_compile_options(${PROJECT_NAME} PRIVATE -Xcudafe "--diag_suppress=esa_on_defaulted_function_ignored")
 
#target_compile_options(${PROJECT_NAME} PRIVATE -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -Xcudafe --diag_suppress=useless_type_qualifiers)
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
 
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROGRESSION_LIBS} cudart)
