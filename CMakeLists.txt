cmake_minimum_required (VERSION 3.22)
project (HARRIS CUDA CXX)

include(FetchContent)

FetchContent_Declare(
  GoogleBenchmark
  URL https://github.com/google/benchmark/archive/v1.8.0.tar.gz
  )

FetchContent_Declare(
  CLI11
  URL https://github.com/CLIUtils/CLI11/archive/v2.3.2.tar.gz
  )

FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/v1.11.0.tar.gz
  )


if (NOT GoogleBenchmark_POPULATED)
  FetchContent_Populate(GoogleBenchmark)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
  add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif()

FetchContent_MakeAvailable(spdlog)
FetchContent_MakeAvailable(CLI11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )
find_package(PNG REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -pedantic -std=c++17 -O3 -DNDEBUG -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_ARCHITECTURES 75 86)


add_library(renderer_cpu)
target_compile_features(renderer_cpu PUBLIC)
target_link_libraries(renderer_cpu PRIVATE spdlog::spdlog)
target_sources(renderer_cpu PRIVATE src/CPU/corners_cpu.cpp)
target_include_directories(renderer_cpu PRIVATE src src/CPU)


add_library(renderer_gpu)
target_compile_features(renderer_gpu PUBLIC)
target_link_libraries(renderer_gpu PRIVATE spdlog::spdlog)
target_sources(renderer_gpu PRIVATE src/GPU/corners_gpu.cu src/GPU/separable_convolution.cu src/GPU/error.cu)
target_include_directories(renderer_gpu PRIVATE src src/GPU)
set_target_properties(renderer_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(render)
target_sources(render PRIVATE src/render.cpp src/read_png.cpp)
target_include_directories(render PRIVATE src)
target_compile_features(render PUBLIC)
target_link_libraries(render PRIVATE PNG::PNG CLI11::CLI11 spdlog::spdlog renderer_cpu renderer_gpu)

add_executable(bench)
target_sources(bench PRIVATE src/bench.cpp src/read_png.cpp)
target_include_directories(bench PRIVATE src)
target_compile_features(bench PUBLIC)
target_link_libraries(bench PRIVATE PNG::PNG spdlog::spdlog benchmark renderer_cpu renderer_gpu)
