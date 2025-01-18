//
//  ActivationFunctions.cpp
//  NeuralNetwork
//
//  Created by Richard Dalley on 2025-01-09.
//
#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <cmath>
#include "activation_functions.h" // Your Matrix class
#include "matrix.h" // Your Matrix class

namespace NeuralNetwork{
        namespace ActivationFunctions {
            // Sigmoid activation function
            float sigmoid(float x) {
                return 1.0f / (1.0f + std::exp(-x));
            }

            // Derivative of sigmoid (for backpropagation)
            float sigmoidDerivative(float x) {
                float s = sigmoid(x);
                return s * (1 - s);
            }

            // ReLU activation function
            float relu(float x) {
                return (x > 0) ? x : 0.0f;
            }

            // Derivative of ReLU (for backpropagation)
            float reluDerivative(float x) {
                return (x > 0) ? 1.0f : 0.0f;
            }
            float tanh(float x) {
                return std::tanh(x);
            }
            float tanhDerivative(float x) {
                float t = tanh(x);
                return 1 - t * t;
            }
            float leakyRelu(float x) {
                return (x > 0) ? x : 0.01f * x;
            }
            float leakyReluDerivative(float x) {
                return (x > 0) ? 1.0f : 0.01f;
            }
            
            // Apply any activation function element-wise to a matrix
            void apply(NeuralNetwork::Matrix<float>& mat, std::function<float(float)> func) {
                for (size_t i = 0; i < mat.getRows(); ++i) {
                    for (size_t j = 0; j < mat.getCols(); ++j) {
                        mat[i][j] = func(mat[i][ j]);
                    }
                }
            }
            
            NeuralNetwork::Matrix<float> applyNew(const NeuralNetwork::Matrix<float>& mat, std::function<float(float)> func) {
                Matrix<float> result(mat.getRows(), mat.getCols());  // Create a new matrix with the same dimensions
                for (size_t i = 0; i < mat.getRows(); ++i) {
                    for (size_t j = 0; j < mat.getCols(); ++j) {
                        result[i][j] = func(mat[i][ j]);  // Apply the function to each element
                    }
                }
                return result;
            }
        };
}

#endif // ACTIVATION_FUNCTIONS_H

