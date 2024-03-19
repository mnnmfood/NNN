
#include <cmath>
#include <iostream>
#include <functional>
#include <costs.h>
#include "eigenFuns.h"
#include "typedefs.h"
#include "cost_funs.h"
#include "layer_activations.h"

inline const std::array<int, 1> dims {0};

Tensor<float, 0> MSE::cost(tmap_t a, 
                tmap_t y, ThreadPoolDevice* device){
    return mse_fun(a, y, device);
}

void MSE::grad(
        tmap_t a,
        tmap_t y,
        tmap_t grad,
        ThreadPoolDevice* device){
    mse_grad_fun(a, y, grad, device);
}

void MSE::act(tmap_t z, 
    tmap_t act, ThreadPoolDevice* device){
    act = z;
}

Tensor<float, 0> CrossEntropy::cost(tmap_t a, 
            tmap_t y, ThreadPoolDevice* device){
    return cross_entropy_fun(a, y, device, _softmax);
    //return -a.binaryExpr(y, std::ref(cross_entropy)).sum();
}

void CrossEntropy::grad(tmap_t a,
    tmap_t y, tmap_t grad,
    ThreadPoolDevice* device){ 
    cross_entropy_grad_fun(a, y, grad, device, _softmax);
    //std::cout << a.chip(0, 1) << "\n\n";
    //std::cout << y.chip(0, 1) << "\n\n";
    //std::cout << grad.chip(0, 1) << "\n\n";
}

void CrossEntropy::act(tmap_t z, 
    tmap_t act, ThreadPoolDevice* device) {
    if (_softmax) {
        softmax_fun(z, act, device);
    }
    else {
        act.device(*device) = z;
    }
}
