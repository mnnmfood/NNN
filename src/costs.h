#ifndef COSTS_H
#define COSTS_H

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <functional>

using Eigen::Matrix;
using Eigen::Dynamic;


template<typename Scalar>
class CostFun
{
public:
    CostFun(){}
    virtual Scalar cost(const Matrix<Scalar, Dynamic, Dynamic>& a,
                        const Matrix<Scalar, Dynamic, Dynamic>& y) = 0;
    virtual Matrix<Scalar, Dynamic, Dynamic> grad(
        const Matrix<Scalar, Dynamic, Dynamic>& a,
        const Matrix<Scalar, Dynamic, Dynamic>& y,
        const Matrix<Scalar, Dynamic, Dynamic>& a_p) = 0;
};

template<typename Scalar>
class MSE: public CostFun<Scalar>
{
public:
    MSE(){}

    Scalar cost(const Matrix<Scalar, Dynamic, Dynamic>& a, 
                const Matrix<Scalar, Dynamic, Dynamic>& y) override{
        return (a-y).colwise().squaredNorm().sum();
    }

    Matrix<Scalar, Dynamic, Dynamic> grad(
        const Matrix<Scalar, Dynamic, Dynamic>& a,
        const Matrix<Scalar, Dynamic, Dynamic>& y,
        const Matrix<Scalar, Dynamic, Dynamic>& a_p) override{
        return (a - y).cwiseProduct(a_p);
    }
};

template<typename Scalar>
Scalar cross_entropy(Scalar a, Scalar y){
    return y*std::log(a) + (1 - y)*std::log(1 - a);
}

template<typename Scalar>
class CrossEntropy: public CostFun<Scalar>
{
public:
    CrossEntropy(){}

    Scalar cost(const Matrix<Scalar, Dynamic, Dynamic>& a, 
                const Matrix<Scalar, Dynamic, Dynamic>& y) override{
        return -a.binaryExpr(y, std::ref(cross_entropy<Scalar>)).sum();
    }

    Matrix<Scalar, Dynamic, Dynamic> grad(
        const Matrix<Scalar, Dynamic, Dynamic>& a,
        const Matrix<Scalar, Dynamic, Dynamic>& y, 
        const Matrix<Scalar, Dynamic, Dynamic>& a_p) override{
        return a - y;
    }
};

#endif