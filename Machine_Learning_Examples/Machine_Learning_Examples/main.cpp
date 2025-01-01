//
//  main.cpp
//  Machine_Learning_Examples
//
//  Created by Ashot Tadevosyan on 01.01.25.
//

#include <iostream>
#include "LinearRegression.h"

int main(int argc, const char * argv[]) {
    
    
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 4, 6, 8, 10}; // y = 2x
    
    double m = 0.0; // Initial slope
    double b = 0.0; // Initial y-intercept
    double alpha = 0.01; // Learning rate
    int iterations = 1000;
    
    std::cout << "Starting Gradient Descent...\n";
    
    GradientDescent(x, y, m, b, alpha, iterations);
    
    // Final parameters
    std::cout << "\nFinal Parameters:\n";
    std::cout << "Slope (m): " << m << "\n";
    std::cout << "Intercept (b): " << b << "\n";
    
    return 0;
}
