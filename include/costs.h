#ifndef COSTS_H
#define COSTS_H

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <functional>

using Eigen::Matrix;
using Eigen::Dynamic;


class CostFun
{
public:
    virtual float cost(const Matrix<float, Dynamic, Dynamic>& a,
                        const Matrix<float, Dynamic, Dynamic>& y) = 0;
    virtual Matrix<float, Dynamic, Dynamic> grad(
        const Matrix<float, Dynamic, Dynamic>& a,
        const Matrix<float, Dynamic, Dynamic>& y) = 0;
    virtual ~CostFun() = default;
};

class MSE: public CostFun
{
public:
    float cost(const Matrix<float, Dynamic, Dynamic>& a, 
                const Matrix<float, Dynamic, Dynamic>& y) override;

    Matrix<float, Dynamic, Dynamic> grad(
        const Matrix<float, Dynamic, Dynamic>& a,
        const Matrix<float, Dynamic, Dynamic>& y) override;
};

float cross_entropy(float a, float y);

class CrossEntropy: public CostFun
{
public:
    float cost(const Matrix<float, Dynamic, Dynamic>& a, 
                const Matrix<float, Dynamic, Dynamic>& y) override;

    Matrix<float, Dynamic, Dynamic> grad(
        const Matrix<float, Dynamic, Dynamic>& a,
        const Matrix<float, Dynamic, Dynamic>& y) override; 
};

#endif