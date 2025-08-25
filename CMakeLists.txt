cmake_minimum_required(VERSION 3.19)
project(NVIDIA_SGEMM_PRACTICE LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# ensure cuda is available
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY 89)

# output directories
set(OUTPUT_DIR "${PROJECT_SOURCE_DIR}/output")
file(MAKE_DIRECTORY ${OUTPUT_DIR})

set(PROFILE_DIR "${PROJECT_SOURCE_DIR}/profile")
file(MAKE_DIRECTORY ${PROFILE_DIR})

find_package(CUDAToolkit REQUIRED COMPONENTS CUDAToolkit::nppc nppi)

# in debug mode, add debug symbols to device code
# this disables most optimizations and kills performance
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# add_compile_options("--ptxas-options=-v")

# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# Configure the source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# generate executables
### npp (benchmark)
add_executable(image_conv_npp src/image_conv_npp.cu src/image_utils.cpp)
set_target_properties(image_conv_npp PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(image_conv_npp ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} CUDA::nppc CUDA::nppif)
### 1a
add_executable(image_conv_kernel_1a src/image_conv_kernel_1a.cu src/image_utils.cpp)
set_target_properties(image_conv_kernel_1a PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(image_conv_kernel_1a ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} CUDA::nppc CUDA::nppif)
### 1b
add_executable(image_conv_kernel_1b src/image_conv_kernel_1b.cu src/image_utils.cpp)
set_target_properties(image_conv_kernel_1b PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(image_conv_kernel_1b ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} CUDA::nppc CUDA::nppif)
### 2
add_executable(image_conv_kernel_2 src/image_conv_kernel_2.cu src/image_utils.cpp)
set_target_properties(image_conv_kernel_2 PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(image_conv_kernel_2 ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} CUDA::nppc CUDA::nppif)
