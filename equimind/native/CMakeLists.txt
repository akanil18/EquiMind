cmake_minimum_required(VERSION 3.20)
project(equimind_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -flto")
set(CMAKE_BUILD_TYPE Release)

# Find pybind11
find_package(pybind11 REQUIRED)

# Build the extension module
pybind11_add_module(equimind_native src/bindings.cpp)

target_include_directories(equimind_native PRIVATE include)

# Install to the equimind package directory
install(TARGETS equimind_native DESTINATION ${CMAKE_INSTALL_PREFIX})
