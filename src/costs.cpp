#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <functional>
#include <costs.h>

using Eigen::Dynamic;

float MSE::cost(const Matrix<float, Dynamic, Dynamic>& a, 
                const Matrix<float, Dynamic, Dynamic>& y){
    return (a-y).colwise().squaredNorm().sum();
}

Matrix<float, Dynamic, Dynamic> MSE::grad(
        const Matrix<float, Dynamic, Dynamic>& a,
        const Matrix<float, Dynamic, Dynamic>& y,
        const Matrix<float, Dynamic, Dynamic>& a_p){
    return (a - y).cwiseProduct(a_p);
}

float cross_entropy(float a, float y){
    return y*std::log(a) + (1 - y)*std::log(1 - a);
}


float CrossEntropy::cost(const Matrix<float, Dynamic, Dynamic>& a, 
            const Matrix<float, Dynamic, Dynamic>& y){
    return -a.binaryExpr(y, std::ref(cross_entropy)).sum();
}

Matrix<float, Dynamic, Dynamic> CrossEntropy::grad(
    const Matrix<float, Dynamic, Dynamic>& a,
    const Matrix<float, Dynamic, Dynamic>& y, 
    const Matrix<float, Dynamic, Dynamic>& a_p){
    return a - y;
}