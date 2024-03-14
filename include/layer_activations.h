#ifndef LAYER_ACT_H
#define LAYER_ACT_H

#include <cmath>
#include <functional>
#include "typedefs.h"

// helper functions
template<typename Scalar>
Scalar tanhc(Scalar z){
    return std::tanh(z);
}

// -- Activation Funcionts

namespace Eigen
{

template<typename Scalar>
void sigmoid_fun(const Tensor<Scalar, 2>& input, Tensor<Scalar, 2>& output, ThreadPoolDevice* device) {
    output.device(*device) = input.unaryExpr(internal::scalar_logistic_op<Scalar>());
}

template<typename Scalar>
void sigmoid_grad_fun(const Tensor<Scalar, 2>& input, Tensor<Scalar, 2>& output, ThreadPoolDevice* device) {
    sigmoid_fun(input, output, device);
    output.device(*device) = (output - output * output).eval();
}

template<typename Scalar>
void tanh_fun(const Tensor<Scalar, 2>& input, Tensor<Scalar, 2>& output, ThreadPoolDevice* device) {
    output.device(*device) = input.unaryExpr(internal::scalar_tanh_op<Scalar>());
}

template<typename Scalar>
void tanh_grad_fun(const Tensor<Scalar, 2>& input, Tensor<float, 2>& output, ThreadPoolDevice* device) {
    tanh_fun(input, output, device);
    output.device(*device) = (Scalar(1.0f) - output * output).eval();
}

template<typename Scalar>
void
softmax_fun(const Tensor<Scalar, 2>& input, Tensor<Scalar, 2>& output, ThreadPoolDevice* device) {
    const DSizes<Index, 1> along_batch{ 0 };
    const DSizes<Index, 2> reshape_dims{ 1, input.dimension(1) };
    const DSizes<Index, 2> batch_dims{ input.dimension(0), 1 };

    output.device(*device)
        = (input - input.maximum(along_batch).eval()
            .reshape(reshape_dims).broadcast(batch_dims)).exp();
    output.device(*device) = (output * output.sum(along_batch)
        .inverse().eval().reshape(reshape_dims)
        .broadcast(batch_dims).eval());
}

template<typename Scalar>
void softmax_grad_fun(const Tensor<Scalar, 2>& input, Tensor<Scalar, 2>& grad, ThreadPoolDevice* device) {
    softmax_fun(input, grad, device);
    grad.device(*device) = (grad - grad * grad).eval();
}

}
#endif
