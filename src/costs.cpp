
#include <cmath>
#include <iostream>
#include <functional>
#include <costs.h>
#include "eigenFuns.h"
#include "typedefs.h"

inline const std::array<int, 1> dims {0};

Tensor<float, 0> MSE::cost(const Tensor<float, 2>& a, 
                const Tensor<float, 2>& y){
    //return (a-y).colwise().squaredNorm().sum();
    return (a-y).reduce(dims, SqNormReducer<float>()).sum();
}

Tensor<float, 2> MSE::grad(
        const Tensor<float, 2>& a,
        const Tensor<float, 2>& y){
    return (a - y);
}

float cross_entropy(float a, float y){
    return y*std::log(a) + (1 - y)*std::log(1 - a);
}


Tensor<float, 0> CrossEntropy::cost(const Tensor<float, 2>& a, 
            const Tensor<float, 2>& y){
    return -a.binaryExpr(y, std::ref(cross_entropy)).sum();
}

Tensor<float, 2> CrossEntropy::grad(
    const Tensor<float, 2>& a,
    const Tensor<float, 2>& y){ 
    return a - y;
}