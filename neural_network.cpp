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
   

// load data using std::getline
void neuralNetwork::loadData() {
    std::ifstream file(dataFile, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open data file: " + dataFile);
    }

    // Reserve memory if the number of rows is specified
    if (this->dataRows > 0) {
        labels.reserve(dataRows);
        data.reserve(dataRows);
    }

    std::string buffer;
    if (this->dataRows > 0) {
        // Optimized processing
        buffer.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());

        size_t start = 0, end = 0;
        int rowCount = 0;

        while (rowCount < dataRows && (end = buffer.find('\n', start)) != std::string::npos) {
            std::string_view line(buffer.data() + start, end - start);
            start = end + 1;

            std::stringstream ss((std::string(line))); // Use double parentheses
            std::string value;
            std::vector<float> row;
            int label;

            // Read the label (first value in the row)
            std::getline(ss, value, ',');
            label = std::stoi(value);
            labels.push_back(label);

            // Read the pixel values
            while (std::getline(ss, value, ',')) {
                row.push_back(std::stof(value) / scalingFactor); // Normalize to [0, 1]
            }
            data.push_back(std::move(row));
            ++rowCount;
        }
    } else {
        // Default processing using std::getline
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value;
            std::vector<float> row;
            int label;

            // Read the label (first value in the row)
            std::getline(ss, value, ',');
            label = std::stoi(value);
            labels.push_back(label);

            // Read the pixel values
            while (std::getline(ss, value, ',')) {
                row.push_back(std::stof(value) / scalingFactor); // Normalize to [0, 1]
            }
            data.push_back(std::move(row));
        }
    }

    // Shuffle data if enabled
    if (this->shuffleData) {
        shuffle();
    }

    // Split data if validation split is enabled
    if (this->validationSplit > 0.0) {
        splitData();
    } else {
        splitIndex = dataRows;
        trainingData = std::move(data);
        trainingLabels = std::move(labels);
    }
}


void neuralNetwork::shuffle(){
    // Create a vector of indices to shuffle
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., data.size() - 1

    // Create a random device and a random number generator
    std::random_device rd;
    std::mt19937 g(rd());

    // Shuffle the indices
    std::shuffle(indices.begin(), indices.end(), g);

    // Use the shuffled indices to reorder data and labels
    std::vector<std::vector<float>> shuffledData;
    std::vector<int> shuffledLabels;

    for (size_t idx : indices) {
        shuffledData.push_back(data[idx]);
        shuffledLabels.push_back(labels[idx]);
    }

    // Replace original data and labels with shuffled versions
    data = std::move(shuffledData);
    labels = std::move(shuffledLabels);
}

void neuralNetwork::splitData(){
    this->splitIndex = static_cast<size_t>(data.size() * (1 - validationSplit));

    trainingData.assign(data.begin(), data.begin() + splitIndex);
    trainingLabels.assign(labels.begin(), labels.begin() + splitIndex);

    validationData.assign(data.begin() + splitIndex, data.end());
    validationLabels.assign(labels.begin() + splitIndex, labels.end());
}

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

neuralNetwork::neuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, float learningRate, float scalingFactor, bool shuffleData, float validationSplit, std::string dataFile, size_t dataRows)
: inputNodes(inputNodes),
  hiddenNodes(hiddenNodes),
  outputNodes(outputNodes),
  learningRate(learningRate),
  scalingFactor(scalingFactor),
  shuffleData(shuffleData),
  validationSplit(validationSplit),
  dataFile(dataFile),
  dataRows(dataRows),
  inputHiddenWeights(hiddenNodes, inputNodes, 0.0f),
  hiddenOutputWeights(outputNodes, hiddenNodes, 0.0f)
{
    // Randomize weights using normal distribution
    if (validationSplit > 0.0){
        this->splitIndex = static_cast<size_t>(dataRows * (1 - validationSplit));
    }
    initializeWeights(inputHiddenWeights, inputNodes);
    initializeWeights(hiddenOutputWeights, hiddenNodes);
}

neuralNetwork neuralNetwork::fromConfigFile(const std::string& configFileLocation) {
    // Local variables to hold configuration
    int inputNodes = 0;
    int hiddenNodes = 0;
    int outputNodes = 0;
    float learningRate = 0.0f;
    float scalingFactor = 0.0f;
    bool shuffleData = true;
    float validationSplit = 0.1;
    size_t dataRows = 0;
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
        scalingFactor = config.at("scaling_factor").get<float>();
        shuffleData = config.at("shuffle_data").get<bool>();
        validationSplit = config.at("validation_split").get<float>();
        dataFile = config.at("data_file").get<std::string>();
        dataRows = config.at("lines_in_file").get<size_t>();

    } catch (const std::exception& e) {
        throw std::runtime_error("Error parsing config file: " + std::string(e.what()));
    }

    // Use the non-static constructor to create the neuralNetwork object
    return neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate, scalingFactor, shuffleData, validationSplit, dataFile, dataRows);
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

    float totalLoss = 0.0f; // To track total training loss
    int correctPredictions = 0; // To track training accuracy

    confidenceChanges = std::vector<float>(digits, 0.0);
    size_t dataSize = trainingData.size();
    // Train the network with the input and target
    if (showProgress){
        std::cout << "\nTraining the network\n" << std::endl;
    }
    for (size_t iter = 0; iter < epochs; ++iter) {
        totalLoss = 0.0f; // Reset total loss for the epoch
        correctPredictions = 0; // Reset correct predictions for the epoch

        for (size_t i = 0; i < dataSize; ++i) {
            //initialize the inuputLayer from the image
            std::vector<float> inputLayer = trainingData[i];
            //initialize the output vector
            std::vector<float> targetLayer(10, 0.1);
            targetLayer[trainingLabels[i]] = 0.99; // One-hot encoding
                                           //train it epoch times
            trainLayer(inputLayer, targetLayer);
            // Get output/confidence for the current input
            Matrix<float> output = forwardPass(inputLayer);
            
            
            // Calculate loss for this input
            std::vector<float> outputLayer = output.toVector(); // Convert Matrix to vector
            float loss = calculateLoss(outputLayer, trainingLabels[i]);
            totalLoss += loss;

            // Determine the predicted digit
            int predictedLabel = getPredictedLabel(outputLayer);
            if (predictedLabel == trainingLabels[i]) {
                ++correctPredictions;
            }
            
            // Update max confidence for the corresponding digit
            for (size_t digit = 0; digit < digits; ++digit) {
                if (outputLayer[digit] > confidenceChanges[digit]) {
                    confidenceChanges[digit] = outputLayer[digit];
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
        
        // Print epoch metrics
        if (showProgress) {
            float averageLoss = totalLoss / dataSize;
            float accuracy = static_cast<float>(correctPredictions) / dataSize * 100.0f;

            std::cout << "\nEpoch " << iter + 1 << "/" << epochs
                      << " - Loss: " << averageLoss
                      << ", Accuracy: " << accuracy << "%\n";
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

float neuralNetwork::calculateLoss(const std::vector<float>& outputLayer, int trueLabel) {
    float loss = 0.0f;
    for (int i = 0; i < outputNodes; ++i) {
        float predicted = outputLayer[i]; // Assume outputLayer stores probabilities
        float target = (i == trueLabel) ? 1.0f : 0.0f; // One-hot target
        loss -= target * log(predicted + 1e-7f); // Add a small value to avoid log(0)
    }
    return loss;
}

int neuralNetwork::getPredictedLabel(const std::vector<float>& outputLayer) {
    return std::distance(outputLayer.begin(),
                         std::max_element(outputLayer.begin(), outputLayer.end()));
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
        << "Input Nodes: " << this->inputNodes <<  std::endl
        << "Hidden Nodes: " << this->hiddenNodes <<  std::endl
        << "Output Nodes: " << this->outputNodes << std::endl
        << "Epochs: " << this->epochs <<  std::endl
        << "Learning Rate: " << std::fixed << std::setprecision(2) << this->learningRate <<  std::endl
        << "Scaling Factor: " << this->scalingFactor << std::endl
        << "Shuffle Data: " << (this->shuffleData ? "true" : "false") << std::endl
        << "Number of Records:" << this->dataRows << std::endl
        << "Validation Split: " << std::fixed << std::setprecision(2)  << this->validationSplit << std::endl
        << "Training Records:" << this->splitIndex << std::endl;
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
