//
//  activation_functions.h
//  NeuralNetwork
//
//  Created by Richard Dalley on 2025-01-16.
//

#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <functional>
#include "matrix.h" // Your Matrix class

namespace ActivationFunctions {
    // Function declarations
    float sigmoid(float x);
    float sigmoidDerivative(float x);
    float relu(float x);
    float reluDerivative(float x);
    float tanh(float x);
    float tanhDerivative(float x);
    float leakyRelu(float x);
    float leakyReluDerivative(float x);

    void apply(Matrix<float>& mat, std::function<float(float)> func);
    Matrix<float> applyNew(const Matrix<float>& mat, std::function<float(float)> func);
};

#endif // ACTIVATION_FUNCTIONS_H
