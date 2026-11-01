cmake_minimum_required(VERSION 3.24)

# Build for the GPU in this machine unless told otherwise (e.g. -DCMAKE_CUDA_ARCHITECTURES=86).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(FastShadowGPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(TIFF REQUIRED)

add_executable(FastShadowGPU src/main.cpp src/shadow.cu)
target_link_libraries(FastShadowGPU PRIVATE TIFF::TIFF)
