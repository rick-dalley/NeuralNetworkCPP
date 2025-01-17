# Primitive C++ Neural Network

This project is a primitive implementation of a neural network written in **C++**, inspired by the ideas presented in _"Make Your Own Neural Network"_ by Tariq Rashid. While the original book uses Python, this project aims to replicate the concepts in C++ as a learning exercise.

---

## Overview

- **Purpose**: This is a simple neural network implementation designed for educational purposes, not a robust production-level neural network library.
- **Training Data**: The neural network uses the [MNIST dataset](https://yann.lecun.com/exdb/mnist/) for training and testing. MNIST is a widely used dataset of handwritten digits (0–9).
- **Output**: After 10 training runs, the network outputs its confidence for each digit (0–9).

---

## Features

- Written in C++ for those interested in exploring neural network concepts in a lower-level language.
- Uses a simple feed-forward, backpropagation-based approach.
- Provides insights into how neural networks learn without relying on external libraries.

---

## Requirements

1. **MNIST Data**:

   - Download the MNIST dataset in CSV format from [Yann LeCun's MNIST page](https://yann.lecun.com/exdb/mnist/).
   - Place the files in a directory accessible to your project (e.g., `Data/mnist`).

2. **C++ Compiler**:

   - A modern C++ compiler (e.g., GCC, Clang, or MSVC) that supports C++11 or later.

3. **Development Environment**:
   - Tested on Xcode, but it should work with any C++ IDE or build tool (e.g., Visual Studio, CMake).

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/rick-dalley/NeuralNetwork.git
   cd NeuralNetwork
   ```
2. Add the MNIST dataset:
   • Place mnist_train.csv (training data) in the Data/mnist directory.

3. The path to the MNIST dataset is hardcoded in `neural_network.h`. Update the following constant with the location of your dataset file:

```neural_network.h
// neural_network.h
const std::string MNIST_PATH = "/absolute/path/to/your/mnist_train.csv";
```

4.  Build and run the project:
    • Compile using your preferred method or IDE.
    • Run the program to train the neural network and view the confidence levels for each digit after a number of training runs (epochs).
    • CMakeList.txt has been included

## Disclaimer

This project is a training exercise only and is not intended to be a comprehensive or robust implementation of a neural network. While you’re welcome to use this code as a starting point for your own learning, please note the following:  
 • It is not optimized for performance or large-scale use.  
 • It is not meant for production environments.  
 • There are alternatives to some modules which may offer optimizations, they have been commented out. You are free to try them to explore any changes they make to the performance, as a means of understanding some options for optimization.  
 • Contributions and feedback are welcome, but the project may lack features found in modern neural network libraries.

## Credits

    •    Book: “Make Your Own Neural Network” by Tariq Rashid.
    •    Dataset: MNIST.

## License

Feel free to use and modify this code for educational purposes. Attribution is appreciated but not required.
