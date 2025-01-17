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
#include "neural_network.h"


// main
int main(int argc, const char * argv[]) {
    (void) argc;
    (void) argv;    
    
    //instantiate the neural network
    auto nn = neuralNetwork::fromConfigFile("/Users/richarddalley/Code/c++/NeuralNetwork/mnist/config.json");
    nn.printConfiguraton();
    //load the images and alter the values from 0-255, to 0 to 1.0
    nn.loadData();
    //train the network with the data
    nn.train(true);
    // Print the initial configuration of the network
    std::cout << "Summary:" << std::endl;
    nn.printSummary();
    
    return 0;
}
