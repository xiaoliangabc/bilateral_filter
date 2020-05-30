cmake_minimum_required(VERSION 2.8)
project(bilateral_filter)

## Compile as C++11
add_definitions(-std=c++11)

## Find macros and libraries
find_package(OpenCV REQUIRED)

## Specify build type
set(CMAKE_BUILD_TYPE "Release")

## Declare a C++ library
add_executable(bilateral_filter main.cc)

## Specify libraries to link a library or executable target against
target_link_libraries(bilateral_filter ${OpenCV_LIBS} )
