//
//  neural_network.h
//  NeuralNetwork
//
//  Created by Richard Dalley on 2025-01-15.
//


#include <sstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <random>
#include <stdexcept>
#include "activation_functions.h"

class neuralNetwork {
    int inputNodes = 0;
    int hiddenNodes = 0;
    int outputNodes = 0;
    size_t epochs = 1;
    std::mt19937 gen; // Random number generator

    float learningRate = 0.3;
    float scalingFactor = 1.0;
    Matrix<float> inputHiddenWeights;
    Matrix<float> hiddenOutputWeights;
    std::string dataFile;
    std::vector<std::vector<float>> data;
    std::vector<int> labels;   
    size_t imageSize = 0;
    size_t digits = 10;
    std::vector<float> confidenceChanges;

public:
    neuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, float learningRate, float scalingFactor, std::string dataFile);
    static neuralNetwork fromConfigFile(const std::string& configFileLocation);
    void initializeWeights(Matrix<float>& matrix, int nodesInPreviousLayer);
    void setLearningRate(float newLearningRate);
    void train(bool showProgress);
    void trainLayer(const std::vector<float>& inputLayer, const std::vector<float>& targetLayer);
    void printWeights();
    void printConfiguraton();
    Matrix<float> forwardPass(std::vector<float>& inputLayer);
    void printOutput(std::vector<float>& inputLayer, int index);
    void printSummary();
    void loadData();

};
