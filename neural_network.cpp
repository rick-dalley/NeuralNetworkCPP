//
//  neural_network.cpp
//  NeuralNetwork
//
//  Created by Richard Dalley on 2025-01-16.
//

#include <fstream>
#include <cmath> // For std::pow
#include <json.hpp>
#include "neural_network.h"
   

// loadMINST using std::getline
void neuralNetwork::loadData() {
    std::ifstream file(dataFile);
    std::string line;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;
        int label;
        
        // Read the label (first value in the row)
        std::getline(ss, value, ',');
        label = std::stoi(value);
        this->labels.push_back(label);
        
        // Read the pixel values
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value) / 255.0f); // Normalize to [0, 1]
        }
        data.push_back(row);
    }
}

// // loadMINST optimized
// // this will save roughly 17% of the time wasted using std::getline - you must know the the file size however
// std::vector<std::vector<float>> loadMNISTOptimized(const std::string& filename, std::vector<int>& labels) {
//     std::ifstream file(filename);
//     if (!file.is_open()) {
//         throw std::runtime_error("Failed to open file: " + filename);
//     }

//     std::string line;
//     std::vector<std::vector<float>> data;

//     // Reserve space to minimize reallocations (optional: adjust based on dataset size)
//     labels.reserve(60000); // Example: for MNIST training set
//     data.reserve(60000);

//     while (std::getline(file, line)) {
//         std::stringstream ss(line);
//         std::string value;

//         // Preallocate row size (for MNIST, it's always 784 pixels)
//         std::vector<float> row;
//         row.reserve(784);

//         // Read the label (first value in the row)
//         std::getline(ss, value, ',');
//         labels.push_back(std::stoi(value));

//         // Read and normalize the pixel values
//         while (std::getline(ss, value, ',')) {
//             row.push_back(std::stof(value) / 255.0f);
//         }

//         data.push_back(std::move(row));
//     }

//     return data;
// }


void printFirstImageInVector(std::vector<std::vector<float>>& images, std::vector<int>& labels){
    // Example: Print first label and image
    std::cout << "Label: " << labels[0] << "\nImage:\n";
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            std::cout << (images[0][i * 28 + j] > 0.5 ? "*" : " ");
        }
        std::cout << "\n";
    }
    
}

neuralNetwork::neuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, float learningRate, std::string dataFile)
: inputNodes(inputNodes),
  hiddenNodes(hiddenNodes),
  outputNodes(outputNodes),
  learningRate(learningRate),
  dataFile(dataFile),
  inputHiddenWeights(hiddenNodes, inputNodes, 0.0f),
  hiddenOutputWeights(outputNodes, hiddenNodes, 0.0f)
{
    // Randomize weights using normal distribution
    initializeWeights(inputHiddenWeights, inputNodes);
    initializeWeights(hiddenOutputWeights, hiddenNodes);
}

neuralNetwork neuralNetwork::fromConfigFile(const std::string& configFileLocation) {
    // Local variables to hold configuration
    int inputNodes = 0;
    int hiddenNodes = 0;
    int outputNodes = 0;
    float learningRate = 0.0f;
    std::string dataFile;

    // Load the configuration
    std::ifstream configFile(configFileLocation);
    if (!configFile.is_open()) {
        throw std::runtime_error("Failed to open config file: " + configFileLocation);
    }

    try {
        // Parse JSON file
        nlohmann::json config;
        configFile >> config;

        // Extract configuration values
        inputNodes = config.at("input_nodes").get<int>();
        hiddenNodes = config.at("hidden_nodes").get<int>();
        outputNodes = config.at("output_classes").get<int>();
        learningRate = config.at("learning_rate").get<float>();
        dataFile = config.at("data_file").get<std::string>();

    } catch (const std::exception& e) {
        throw std::runtime_error("Error parsing config file: " + std::string(e.what()));
    }

    // Use the non-static constructor to create the neuralNetwork object
    return neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate, dataFile);
}

void neuralNetwork::initializeWeights(Matrix<float>& matrix, int nodesInPreviousLayer) {
    std::normal_distribution<float> dist(0.0f, std::pow(nodesInPreviousLayer, -0.5f));

    for (size_t i = 0; i < matrix.getRows(); ++i) {
        for (size_t j = 0; j < matrix.getCols(); ++j) {
            matrix[i][j] = dist(gen);
        }
    }
}

void neuralNetwork::setLearningRate(float newLearningRate) {
    this->learningRate = newLearningRate;
}


void neuralNetwork::train(bool showProgress){
   confidenceChanges = std::vector<float>(digits, 0.0);
    imageSize = data.size();
    // Train the network with the input and target
    if (showProgress){
        std::cout << "\nTraining the network\n" << std::endl;
    }
    for (size_t iter = 0; iter < epochs; ++iter) {

        for (size_t i = 0; i < imageSize; ++i) {
            //initialize the inuputLayer from the image
            std::vector<float> inputLayer = data[i];
            //initialize the output vector
            std::vector<float> targetLayer(10, 0.1);
            targetLayer[labels[i]] = 0.99; // One-hot encoding
                                           //train it epoch times
            trainLayer(inputLayer, targetLayer);
            // Get output/confidence for the current input
            Matrix<float> output = forwardPass(inputLayer);
            
            // Update max confidence for the corresponding digit
            for (size_t digit = 0; digit < digits; ++digit) {
                if (output[digit][0] > confidenceChanges[digit]) {
                    confidenceChanges[digit] = output[digit][0];
                }
            }
            // if the user wants to show the progress of the training
            if (showProgress) { //set to false to train faster
                if (i > 0){
                    size_t prog = (iter * i) + i;
                    if (prog % 1000 == 0) {
                        std::cout << "." << std::flush;
                        if (prog % 10000 == 0) {
                            std::cout << " " << std::flush; 
                        }
                    }
                } else {
                     std::cout << "Progress: " << std::flush;
                }
            }
        }
    }
    if (showProgress){
        std::cout << std::endl;
    }
}

void neuralNetwork::trainLayer(const std::vector<float>& inputLayer, const std::vector<float>& targetLayer) {
    Matrix<float> inputs(inputLayer);
    Matrix<float> targets(targetLayer);

    // Make a forward pass - computing hidden and final outputs
    Matrix<float> hiddenInputs = inputHiddenWeights * inputs;
    Matrix<float> hiddenOutputs = ActivationFunctions::applyNew(hiddenInputs, ActivationFunctions::sigmoid);

    Matrix<float> finalInputs = hiddenOutputWeights * hiddenOutputs;
    Matrix<float> finalOutputs = ActivationFunctions::applyNew(finalInputs, ActivationFunctions::sigmoid);

    // Calculate the output errors
    Matrix<float> outputErrors = targets - finalOutputs;
    Matrix<float> hiddenErrors = hiddenOutputWeights.transpose() * outputErrors;

    // Update weights for hidden-to-output
    Matrix<float> outputGradients = ActivationFunctions::applyNew(finalOutputs, [](float x) { return x * (1.0f - x); });
    Matrix<float> scaledOutputErrors = outputErrors * outputGradients;
    Matrix<float> weightDeltaOutput = scaledOutputErrors * hiddenOutputs.transpose();
    weightDeltaOutput = weightDeltaOutput * learningRate;
    hiddenOutputWeights += weightDeltaOutput;

    // Update weights for input-to-hidden
    Matrix<float> hiddenGradients = ActivationFunctions::applyNew(hiddenOutputs, [](float x) { return x * (1.0f - x); });
    Matrix<float> scaledHiddenErrors = hiddenErrors * hiddenGradients;
    Matrix<float> weightDeltaInput = scaledHiddenErrors * inputs.transpose();
    weightDeltaInput = weightDeltaInput * learningRate;
    inputHiddenWeights += weightDeltaInput;
}

void neuralNetwork::printWeights() {
    std::cout << "Randomized Input Weight Matrix:\n";
    inputHiddenWeights.print();
    std::cout << "Randomized Output Weight Matrix:\n";
    hiddenOutputWeights.print();
}

void neuralNetwork::printSummary(){
        // Print the training summary
        std::cout << "Confidence vector:\n";
        for (size_t digit = 0; digit < digits; ++digit) {
            std::cout << "Digit " << digit << ": " << std::fixed << std::setprecision(2) << confidenceChanges[digit] << " ";
        }
        std::cout << "\n";

}

void neuralNetwork::printConfiguraton() {
    std::stringstream ss;
    ss << "Neural Network\n"
       << "Input Nodes: " << this->inputNodes << "\n"
       << "Hidden Nodes: " << this->hiddenNodes << "\n"
       << "Output Nodes: " << this->outputNodes << "\n"
       << "Epochs: " << this->epochs << "\n"
       << "Learning Rate: " << std::fixed << std::setprecision(2) << this->learningRate << "\n";
    std::cout << ss.str();
}

Matrix<float> neuralNetwork::forwardPass(std::vector<float>& inputLayer) {
    Matrix<float> inputs(inputLayer);
    Matrix<float> hiddenInputs = inputHiddenWeights * inputs;
    Matrix<float> hiddenOutputs = ActivationFunctions::applyNew(hiddenInputs, ActivationFunctions::sigmoid);
    Matrix<float> finalInputs = hiddenOutputWeights * hiddenOutputs;
    Matrix<float> finalOutputs = ActivationFunctions::applyNew(finalInputs, ActivationFunctions::sigmoid);

    return finalOutputs;
}

void neuralNetwork::printOutput(std::vector<float>& inputLayer, int index) {
    Matrix<float> output = forwardPass(inputLayer);
    std::cout << "\nOutput nodes for " << index << ":" << std::endl;
    output.print();
}
