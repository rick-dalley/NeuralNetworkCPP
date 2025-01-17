# Specify the minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Set the project name and specify that we are using C++
project(NeuralNetwork LANGUAGES CXX)

# Set the C++ standard to use
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable for your project
add_executable(nn
    main.cpp
    neural_network.cpp
    activation_functions.cpp
)

target_include_directories(nn PRIVATE ${CMAKE_SOURCE_DIR}/include)