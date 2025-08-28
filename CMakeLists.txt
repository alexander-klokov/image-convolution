cmake_minimum_required(VERSION 3.19)
project(NVIDIA_SGEMM_PRACTICE LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# ensure cuda is available
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY 89) # for NVIDIA GeForce RTX 4060 Laptop GPU, Ada Lovelace architecture

# output directories
set(OUTPUT_DIR "${PROJECT_SOURCE_DIR}/output")
file(MAKE_DIRECTORY ${OUTPUT_DIR})

set(PROFILE_DIR "${PROJECT_SOURCE_DIR}/profile")
file(MAKE_DIRECTORY ${PROFILE_DIR})

find_package(CUDAToolkit REQUIRED COMPONENTS CUDAToolkit::nppc nppi)

# static library
add_library(image_utils src/image_utils.cpp)

# configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# configure the source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

macro(add_cuda_executable name)
    add_executable(${name} src/${name}.cu)
    set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
    target_link_libraries(${name} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} CUDA::nppc CUDA::nppif image_utils)
endmacro()

# generate executables
### npp (benchmark)
add_executable(image_conv_npp src/image_conv_npp.cu)
set_target_properties(image_conv_npp PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(image_conv_npp ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} CUDA::nppc CUDA::nppif image_utils)
### kernels
add_cuda_executable(image_conv_kernel_1a)
add_cuda_executable(image_conv_kernel_1b)
add_cuda_executable(image_conv_kernel_2)
add_cuda_executable(image_conv_kernel_3)
add_cuda_executable(image_conv_kernel_4)
