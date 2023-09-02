cmake_minimum_required(VERSION 3.22)

SET (CMAKE_CXX_COMPILER "/usr/bin/clang++" CACHE STRING "C++ compiler" FORCE)

project (netzwerk)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Wall -Wextra")

find_package(Threads REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# Note: `TensorNetwork` should come before `Optimizer`.
file(GLOB INCLUDE_FILES
  "include/TensorNetwork.hpp"
  "include/Optimizer.hpp"
  "include/BitSet.hpp"
  "include/Util.hpp"
  "include/Common.hpp"
)

file(GLOB SRC_FILES "include/*.cpp")

# Collect all.
file(GLOB ALL_FILES ${INCLUDE_FILES} ${SRC_FILES})

# Build the shared library.
add_library(netzwerk SHARED ${ALL_FILES} netzwerk.cc)

# Build the main for testing.
add_executable(main ${ALL_FILES} main.cc)