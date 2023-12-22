#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <functional>

using Eigen::Matrix;
using Eigen::Dynamic;


template<typename Scalar>
class ActivationFun
{
public:
    ActivationFun(){}
    virtual Matrix<Scalar, Dynamic, Dynamic> activation(const Matrix<Scalar, Dynamic, Dynamic>& z) = 0;
    virtual Matrix<Scalar, Dynamic, Dynamic> activation_prime(const Matrix<Scalar, Dynamic, Dynamic>& z) = 0;
};

template<typename Scalar>
Scalar logistic(Scalar z){
    return Dynamic / (Dynamic + std::exp(-z));
}

template<typename Scalar> 
Scalar logistic_prime(Scalar z){
    return (1.0 - logistic(z))*logistic(z);
}

template<typename Scalar>
class Logistic: public ActivationFun<Scalar>
{
public:
    Logistic(){};

    Matrix<Scalar, Dynamic, Dynamic> activation(const Matrix<Scalar, Dynamic, Dynamic>& z) override{
        return z.unaryExpr(std::ref(logistic<Scalar>));
    }

    Matrix<Scalar, Dynamic, Dynamic> activation_prime(const Matrix<Scalar, Dynamic, Dynamic>& z) override{
        return z.unaryExpr(std::ref(logistic_prime<Scalar>));
    }
};

#endif