//
//  main.cpp
//  Deep_Learning_Examples
//
//  Created by Ashot Tadevosyan on 01.01.25.
//
//Deep Learning example in C++ that uses a simple feedforward neural
//network. I'll manually implement a basic neural network to demonstrate
//concepts like forward propagation and training using backpropagation.
//
//Example: Feedforward Neural Network for XOR Problem
//
//The XOR problem is a classic problem in deep learning, where the neural network learns to solve the XOR logic gate.


#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>


double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x)
{
    return x * (1.0 - x);
}


std::vector<std::vector<double>> inputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };

std::vector<double> outputs = {0, 1, 1, 0};
int main(int argc, const char * argv[])
{
    std::srand(static_cast<unsigned>(std::time(0)));
    
    double weight1 = (std::rand() % 100) / 100.0;
    
    double weight2 = (std::rand() % 100) / 100.0;
    
    double bias1 = (std::rand() % 100) / 100.0;
    
    double weightOut = (std::rand() % 100) / 100.0;
    
    double biasOut = (std::rand() % 100) / 100.0;
    
    double learningRate = 0.1;
    int epochs = 10000;
    
    
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double totalError = 0.0;
        for (int i = 0; i < inputs.size(); ++i) {
            // Forward propagation
            double x1 = inputs[i][0];
            double x2 = inputs[i][1];
            double target = outputs[i];
            double hiddenNet = weight1 * x1 + weight2 * x2 + bias1;
            
            double hiddenOutput = sigmoid(hiddenNet);
            double outputNet = weightOut * hiddenOutput + biasOut;
            double output = sigmoid(outputNet);
            // Error calculation
            double error = 0.5 * pow((target - output), 2);
            totalError += error;
            double outputError = (output - target) * sigmoidDerivative(output);
            double hiddenError = outputError * weightOut * sigmoidDerivative(hiddenOutput);
            
            weightOut -= learningRate * outputError * hiddenOutput;
            biasOut -= learningRate * outputError;
            weight1 -= learningRate * hiddenError * x1;
            weight2 -= learningRate * hiddenError * x2;
            bias1 -= learningRate * hiddenError;
        }
        
        if (epoch % 1000 == 0)
        {
            std::cout << "Epoch " << epoch << ", Error: " << totalError << "\n";
        }
    }
    
    std::cout << "\nTrained Neural Network Results:\n";
    
    for (int i = 0; i < inputs.size(); ++i)
    {
        double x1 = inputs[i][0];
        double x2 = inputs[i][1];
        
        double hiddenNet = weight1 * x1 + weight2 * x2 + bias1;
        double hiddenOutput = sigmoid(hiddenNet);
        
        double outputNet = weightOut * hiddenOutput + biasOut;
        double output = sigmoid(outputNet);
        
        std::cout << "Input: (" << x1 << ", " << x2 << ") -> Output: " << output << "\n";
    }
    
    return 0;
}
