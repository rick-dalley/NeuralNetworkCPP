//
//  main.cpp
//  NeuralNetwork
//
//  Created by Richard Dalley on 2025-01-09.
//
#include <sstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <stdexcept>
#include "model.h"

using namespace NeuralNetwork;

// main
int main(int argc, const char * argv[]) {
    (void) argc;
    (void) argv;    
    
    //instantiate the neural network
    auto model = Model::fromConfigFile("/Users/richarddalley/Code/c++/NeuralNetwork/mnist/config.json");
    model.printConfiguraton();
    //load the images and alter the values from 0-255, to 0 to 1.0
    model.loadData();
    //train the network with the data
    model.train(true);
    // Print the initial configuration of the network
    std::cout << "Summary:" << std::endl;
    model.printSummary();
    
    return 0;
}
