//
//  LinearRegression.h
//  Machine_Learning_Examples
//
//  Created by Ashot Tadevosyan on 01.01.25.
//
#ifndef LinearRegression
#define LinearRegression

double ComputeCost(const std::vector<double>& x, const std::vector<double>& y, double m, double b);

void GradientDescent(const std::vector<double>& x,
                     const std::vector<double>& y,
                     double& m, double& b, double alpha, int iterations);

#endif
