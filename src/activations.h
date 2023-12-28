#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <functional>

using Eigen::Matrix;
using Eigen::Vector;
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
    return 1.0 / (1.0 + std::exp(-z));
}

template<typename Scalar> 
Scalar logistic_prime(Scalar z){
    Scalar log = 1.0 / (1.0 + std::exp(-z));
    return (1.0 - log)*log;
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

template<typename Scalar>
Scalar tanhc(Scalar z){
    return std::tanh(z);
}

template<typename Scalar> 
Scalar tanh_prime(Scalar z){
    Scalar th = std::tanh(z);
    return 1.0 - th * th;
}



template<typename Scalar>
class Tanh: public ActivationFun<Scalar>
{
public:
    Tanh(){};

    Matrix<Scalar, Dynamic, Dynamic> activation(const Matrix<Scalar, Dynamic, Dynamic>& z) override{
        return z.unaryExpr(std::ref(tanhc<Scalar>));
    }

    Matrix<Scalar, Dynamic, Dynamic> activation_prime(const Matrix<Scalar, Dynamic, Dynamic>& z) override{
        return z.unaryExpr(std::ref(tanh_prime<Scalar>));
    }
};

template<typename Scalar>
class usrExp
{
    Scalar m{};
public:
    usrExp(Scalar i) :m{i}{}
    Scalar operator()(Scalar z) const{
        return std::exp(z - m);
    }
};

template<typename Scalar>
Scalar usrLog(Scalar z){
    return std::log(z);
}

template<typename Scalar>
class SoftMax: public ActivationFun<Scalar>
{
    public:
    SoftMax(){}

    Matrix<Scalar, Dynamic, Dynamic> activation(const Matrix<Scalar, Dynamic, Dynamic>& z) override{
        Matrix<Scalar, Dynamic, Dynamic> temp_max = z.colwise().maxCoeff();
        Matrix<Scalar, Dynamic, Dynamic> temp_exp{z.rows(), z.cols()};

        for(int i{0}; i < z.cols(); i++){
            temp_exp.col(i) = z.col(i).unaryExpr(usrExp<Scalar>(temp_max(i)));
        }

        Vector<Scalar, Dynamic> temp_sum = temp_exp.colwise().sum().unaryExpr(std::ref(usrLog<Scalar>));
        Matrix<Scalar, Dynamic, Dynamic> res(z.rows(), z.cols());
        for(int i{0}; i < res.cols(); i++){
            res.col(i) = z.col(i).unaryExpr(usrExp<Scalar>(temp_sum(i) + temp_max(i)));
        }

        return res;
    }

    Matrix<Scalar, Dynamic, Dynamic> activation_prime(const Matrix<Scalar, Dynamic, Dynamic>& z) override{
        Matrix<Scalar, Dynamic, Dynamic> temp_max = z.colwise().maxCoeff();
        Matrix<Scalar, Dynamic, Dynamic> temp_exp(z.rows(), z.cols());

        for(int i{0}; i < z.cols(); i++){
            temp_exp.col(i) = z.col(i).unaryExpr(usrExp<Scalar>(temp_max(i)));
        }

        Vector<Scalar, Dynamic> temp_sum = temp_exp.colwise().sum().unaryExpr(std::ref(usrLog<Scalar>));
        Matrix<Scalar, Dynamic, Dynamic> res(z.rows(), z.cols());
        for(int i{0}; i < res.cols(); i++){
            res.col(i) = z.col(i).unaryExpr(usrExp<Scalar>(temp_sum(i) + temp_max(i)));
        }
        return  res - res.cwiseProduct(res); 
    }

};

#endif