cmake_minimum_required(VERSION 3.28)
project(numpyProj)

set(CMAKE_CXX_STANDARD 17)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
# Set the path to the pybind11 CMake configuration
set(pybind11_DIR "C:/Users/schif/AppData/Local/Programs/Python/Python312/Lib/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 CONFIG REQUIRED)

# Use Python_INCLUDE_DIRS, Python_LIBRARIES, and Python_NumPy_INCLUDE_DIRS variables
include_directories(${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS} ${pybind11_DIR} "C:/Users/schif/AppData/Local/Programs/Python/Python312/Lib/site-packages/pybind11/include")

# Create a shared library for the Python extension module
pybind11_add_module(numpyProj numpyProj/numpyProj.cpp)

# Set the output directory to the numpyProj package directory
set_target_properties(numpyProj PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/numpyProj"
)
