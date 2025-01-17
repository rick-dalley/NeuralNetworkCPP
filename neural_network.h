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

    float learningRate = 0.3;
    float scalingFactor = 1.0;
    bool shuffleData = true;
    float validationSplit = 0.1;
    size_t dataRows = 0;
    size_t splitIndex = 0;
    std::mt19937 gen; // Random number generator

    Matrix<float> inputHiddenWeights;
    Matrix<float> hiddenOutputWeights;
    std::string dataFile;
    std::vector<std::vector<float>> data;
    std::vector<std::vector<float>> trainingData;
    std::vector<std::vector<float>> validationData;
    std::vector<int> labels;   
    std::vector<int> trainingLabels;   
    std::vector<int> validationLabels;   
    size_t digits = 10;
    std::vector<float> confidenceChanges;
    float calculateLoss(const std::vector<float>& outputLayer, int trueLabel);
    int getPredictedLabel(const std::vector<float>& outputLayer);
    void trainLayer(const std::vector<float>& inputLayer, const std::vector<float>& targetLayer);
    void shuffle();
    void splitData();
    void initializeWeights(Matrix<float>& matrix, int nodesInPreviousLayer);

public:
    neuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, float learningRate, float scalingFactor, bool shuffleData, float validationSplit, std::string dataFile, size_t dataRows);
    static neuralNetwork fromConfigFile(const std::string& configFileLocation);
    void setLearningRate(float newLearningRate);
    void train(bool showProgress);
    void printWeights();
    void printConfiguraton();
    Matrix<float> forwardPass(std::vector<float>& inputLayer);
    void printOutput(std::vector<float>& inputLayer, int index);
    void printSummary();
    void loadData();

};
