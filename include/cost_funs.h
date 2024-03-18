#ifndef COST_FUNS_H
#define COST_FUNS_H

#include "typedefs.h"
#include "layer_activations.h"

namespace Eigen
{

template<typename ArgType1, typename ArgType2>
Tensor<typename internal::traits<ArgType1>::Scalar, 0>
mse_fun(ArgType1& input, ArgType2& output, 
    ThreadPoolDevice* device){ 
    typedef internal::traits<ArgType1>::Scalar Scalar;
    const DSizes<Index, 1> along_input{ 0 };
    Tensor<Scalar, 2> temp(input.dimensions());
    temp.device(*device) = input - output;
    return (temp * temp).sum(along_input)
        .sqrt().sum(along_input);
}

template<typename ArgType1, typename ArgType2, 
    typename ArgType3>
void
mse_grad_fun(ArgType1& input, ArgType2& output, 
    ArgType3& grad, ThreadPoolDevice* device) {
    grad.setConstant(0.0f);
    //grad.device(*device) = output - input;
    grad.device(*device) = input - output;
}

template<typename ArgType1, typename ArgType2> 
Tensor<typename internal::traits<ArgType1>::Scalar, 0>
cross_entropy_fun(ArgType1& input, ArgType2& output, 
    ThreadPoolDevice* device, bool with_softmax=true) {
    typedef internal::traits<ArgType1>::Scalar Scalar;

    if (with_softmax) {
        Tensor<Scalar, 2> softmax_input(input.dimensions());
        softmax_fun(input, softmax_input, device);
		auto shifted_input = Scalar(1.0f) - softmax_input;
		auto shifted_output = Scalar(1.0f) - output;
		return -(output * softmax_input.unaryExpr(internal::scalar_log_op<Scalar>()) +
			shifted_output * shifted_input.unaryExpr(internal::scalar_log_op<Scalar>())).sum();
    }

    auto shifted_input = Scalar(1.0f) - input;
    auto shifted_output = Scalar(1.0f) - output;
    return -(output * input.unaryExpr(internal::scalar_log_op<Scalar>()) +
        shifted_output * shifted_input.unaryExpr(internal::scalar_log_op<Scalar>())).sum();
}

template<typename ArgType1, typename ArgType2, 
    typename ArgType3>
auto
cross_entropy_grad_fun(ArgType1& input, ArgType2& output, ArgType3& grad,
    ThreadPoolDevice* device, bool with_softmax = true) {
    typedef internal::traits<ArgType1>::Scalar Scalar;
    grad.setConstant(0.0f);
    if (with_softmax) {
        grad.device(*device) = input - output;
    }
    else {
        auto shifted_input = Scalar(1.0f) - input;
        auto shifted_output = Scalar(1.0f) - output;
        grad.device(*device) = shifted_output * shifted_input.inverse() -
            output * input.inverse();
    }
}

}

#endif
