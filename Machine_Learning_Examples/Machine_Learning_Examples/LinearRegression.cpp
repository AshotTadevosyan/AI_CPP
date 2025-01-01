//
//  LinearRegression.cpp
//  Machine_Learning_Examples
//
//  Created by Ashot Tadevosyan on 01.01.25.
//
#include <iostream>
#include "vector"
#include "cmath"
#include "LinearRegression.h"


double ComputeCost(const std::vector<double>& x, const std::vector<double>& y, double m, double b)
{
    double cost = 0.0;
    unsigned long n = x.size();
    
    for (int i = 0; i < n; ++i)
    {
        double prediction = m * x[i] + b;
        cost += pow((prediction - y[i]), 2);
    }
    
    return cost / (2 * n);
}

void GradientDescent(const std::vector<double>& x,
                     const std::vector<double>& y,
                     double& m, double& b, double alpha, int iterations)
{
    unsigned long n = x.size();
    for (int i = 0; i < iterations; ++i)
    {
        double dm = 0.0; // Gradient for m
        double db = 0.0; // Gradient for b
        
        for (int j = 0; j < n; ++j)
        {
            double prediction = m * x[j] + b;
            dm += (prediction - y[j]) * x[j];
            db += (prediction - y[j]);
        }
        
        m -= alpha * dm / n;
        b -= alpha * db / n;

        if (i % 100 == 0)
        {
            std::cout << "Iteration " << i << ": Cost = " << ComputeCost(x, y, m, b) << "\n";
        }
        
    }
}
