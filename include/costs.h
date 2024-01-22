#ifndef COSTS_H
#define COSTS_H

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <functional>
#include "typedefs.h"

using Eigen::Matrix;
using Eigen::Dynamic;


class CostFun
{
public:
    virtual Tensor<float, 0> cost(const Tensor<float, 2>& a,
                        const Tensor<float, 2>& y) = 0;
    virtual Tensor<float, 2> grad(
        const Tensor<float, 2>& a,
        const Tensor<float, 2>& y) = 0;
    virtual ~CostFun() = default;
};

class MSE: public CostFun
{
public:
    Tensor<float, 0> cost(const Tensor<float, 2>& a, 
                const Tensor<float, 2>& y) override;

    Tensor<float, 2> grad(
        const Tensor<float, 2>& a,
        const Tensor<float, 2>& y) override;
};

float cross_entropy(float a, float y);

class CrossEntropy: public CostFun
{
public:
    Tensor<float, 0> cost(const Tensor<float, 2>& a, 
                const Tensor<float, 2>& y) override;

    Tensor<float, 2> grad(
        const Tensor<float, 2>& a,
        const Tensor<float, 2>& y) override; 
};

#endif