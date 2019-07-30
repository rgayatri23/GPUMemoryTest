cmake_minimum_required (VERSION 3.12 FATAL_ERROR)
project (GPUMemoryTests LANGUAGES CXX CUDA VERSION 0.0.1)
find_package(CUDA 10.1 REQUIRED)

cuda_add_executable(cmake_axpy.ex axpy.cu)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(STATUS "WARNING - DO not build in the same dir as the source")
else()
  message(STATUS "Building the binary files in ${CMAKE_BINARY_DIR}")
endif()

option(TIMEMORY_PROFILE "Enable timemory" OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)

target_include_directories(cmake_axpy.ex PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)

target_compile_definitions(cmake_axpy.ex PUBLIC $<$<BOOL:${CHECK_CORRECTNESS}>:VERIFY_GPU_CORRECTNESS>)
target_compile_definitions(cmake_axpy.ex PUBLIC $<$<BOOL:${PINNED_MEMORY}>:USE_PINNED_MEMORY>)
target_compile_definitions(cmake_axpy.ex PUBLIC $<$<BOOL:${MANAGED_MEMORY}>:USE_MANAGED_MEMORY>)
target_compile_definitions(cmake_axpy.ex PUBLIC $<$<BOOL:${ZERO_COPY}>:USE_ZERO_COPY>)
target_compile_definitions(cmake_axpy.ex PUBLIC $<$<BOOL:${HOST_PAGEABLE_AND_DEVICE_MEMORY}>:USE_HOST_PAGEABLE_AND_DEVICE_MEMORY>)

if(TIMEMORY_PROFILE)
  find_package(TiMemory REQUIRED COMPONENTS headers cuda)
  target_compile_definitions(timemory_profile INTERFACE USE_TIMEMORY)
  target_include_directories(timemory_profile INTERFACE ${TiMemory_INCLUDE_DIR})
target_link_libraries(timemory_profile INTERFACE timemory)
endif()
