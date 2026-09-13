cmake_minimum_required(VERSION 3.18)
project(gkt_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pybind11_ok)
if(NOT _pybind11_ok EQUAL 0)
  message(FATAL_ERROR "pybind11 not found. pip install pybind11")
endif()
find_package(pybind11 CONFIG REQUIRED)

set(GKT_SOURCES
  src/graph.cpp
  src/engine.cpp
  src/mcts.cpp
  src/selfplay.cpp
  src/torch_net.cpp
  bindings/gkt_native.cpp)

# Optional CMake-only; `python cpp/build.py` (setuptools) never sets this.
option(GKT_WITH_TORCH "Link LibTorch for TorchScript inference" OFF)

if(GKT_WITH_TORCH)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _torch_ok)
  if(_torch_ok EQUAL 0 AND NOT TORCH_PREFIX STREQUAL "")
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_PREFIX}")
    find_package(Torch QUIET)
  endif()
endif()

pybind11_add_module(gkt_native ${GKT_SOURCES})
target_include_directories(gkt_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(Torch_FOUND)
  target_compile_definitions(gkt_native PRIVATE GKT_WITH_TORCH)
  target_link_libraries(gkt_native PRIVATE ${TORCH_LIBRARIES})
  if(MSVC)
    target_compile_options(gkt_native PRIVATE /MD)
  endif()
  message(STATUS "gkt_native: LibTorch enabled")
else()
  message(STATUS "gkt_native: building without LibTorch (Python predict_batch callback only)")
endif()

if(MSVC)
  target_compile_options(gkt_native PRIVATE /EHsc /bigobj)
endif()
