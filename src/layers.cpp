
#include "layers.h"
#include <Eigen/Dense>
#include <iostream>
#include "eigenFuns.h"
#include "typedefs.h"

inline const array<int, 1> dims_colwise {0};
inline const array<int, 1> dims_rowwise {1};

scalar_comparer_op<float> min_comparer([](float x, float y)->float {return x < y ? x: y;});
scalar_comparer_op<float> max_comparer([](float x, float y)->float {return x > y ? x: y;});

// Util class for weight initialization
class NormalSample
{
    std::normal_distribution<float> nd;
public:
    NormalSample() = delete;
    NormalSample(float _mean, float _std): nd(_mean, _std){}
    float operator()(float z) {return static_cast<float>(nd(gen));}
};

// helper functions
float logistic(float z){
    return 1.0f / (1.0f + std::exp(-z));
}

float logistic_prime(float z){
    float log = 1.0 / (1.0 + std::exp(-z));
    return (1.0 - log)*log;
}

float tanhc(float z){
    return std::tanh(z);
}

float tanh_prime(float z){
    float th = std::tanh(z);
    return 1.0 - th * th;
}

float usrLog(float z){
    return std::log(z);
}

class usrExp
{
    float m{};
public:
    usrExp(float i) :m{i}{}
    float operator()(float z) const{
        return std::exp(z - m);
    }
};

 // Generic Layer
Layer::Layer(size_t size)
    :_size{size}
{

}
Layer* Layer::next(){return _next;}
Layer* Layer::prev(){return _prev;}

// Input Layer
InputLayer::InputLayer(size_t size)
    :Layer{size}
{
}

void InputLayer::init(size_t n_samples){
    _act = Tensor<float, 2>(_size, n_samples);
}

void InputLayer::fwd(const Tensor<float, 2>& input){
    _act = input;
}

Tensor<float, 2> InputLayer::get_act(){
    return _act;
}

Tensor<float, 2> InputLayer::get_grad(){
    return _act;
}


// Fully connected layer
FCLayer::FCLayer(size_t size)
    :Layer{size}
{
}

void FCLayer::initParams(size_t prev_size){
    NormalSample sampleFun(0.0f, 1.0f / std::sqrt(
        static_cast<float>(prev_size)
    ));

    _weights = Tensor<float, 2>(_size, prev_size).unaryExpr(std::ref(sampleFun));
    _biases = Tensor<float, 1>(_size).unaryExpr(std::ref(sampleFun));
}

void FCLayer::init(size_t n_samples){
    _act = Tensor<float, 2>(_size, n_samples); 
    _grad = Tensor<float, 2>(_size, n_samples); 
    _winputs = Tensor<float, 2>(_size, n_samples);
    _nabla_b = Tensor<float, 2>(_size, n_samples); 
    _nabla_w = Tensor<float, 2>(_size, n_samples); 
}

Tensor<float, 2> FCLayer::get_act(){return _act;}
Tensor<float, 2> FCLayer::get_grad(){return _grad;}

void FCLayer::fwd(){
    assert(_prev != nullptr);
    //_winputs = (_weights * _prev->get_act()).colwise() + _biases ;
    _winputs = vecSum(_weights * _prev->get_act(), _biases);
    _act = act(_winputs);
}

void FCLayer::bwd(const Tensor<float, 2>& cost_grad){
    assert(_next == nullptr);
    _nabla_b = cost_grad.cwiseProduct(grad_act(_winputs));
    //_nabla_w = _nabla_b * _prev->get_act().transpose();
    _nabla_w = _nabla_b * transposed(_prev->get_act());
    //_grad = _weights.transpose() * _nabla_b;
    _grad = transposed(_weights) * _nabla_b;
}

void FCLayer::bwd(){
    assert(_next != nullptr);
    _nabla_b = _next->get_grad().cwiseProduct(grad_act(_winputs));
    //_nabla_w = _nabla_b * _prev->get_act().transpose();
    _nabla_w = _nabla_b * transposed(_prev->get_act());
    //_grad = _weights.transpose() * _nabla_b;
    _grad = transposed(_weights) * _nabla_b;
}

// TODO: updating method should be specific to optimization strategy,
// this should not be here
void FCLayer::update(float rate, float mu, float size){
    _weights = (1 - rate * mu / size) * _weights.eval()- (rate / size) * _nabla_w;
    //_biases -= (rate / size) * (_nabla_b.rowwise().sum());
    _biases -= (rate / size) * (_nabla_b.sum(dims_rowwise));
}

// Sigmoid layer
SigmoidLayer::SigmoidLayer(size_t size) :FCLayer{size}{}

Tensor<float, 2> SigmoidLayer::act(const Tensor<float, 2>& z){
    return z.unaryExpr(std::ref(logistic));
}

Tensor<float, 2> SigmoidLayer::grad_act(const Tensor<float, 2>& z){
    return z.unaryExpr(std::ref(logistic_prime));
}

// Tanh Layer
Tensor<float, 2> TanhLayer::act(const Tensor<float, 2>& z){
    return z.unaryExpr(std::ref(tanhc));
}

Tensor<float, 2> TanhLayer::grad_act(const Tensor<float, 2>& z){
    return z.unaryExpr(std::ref(tanh_prime));
}

// SoftMax Layer
Tensor<float, 2> SoftMaxLayer::act(const Tensor<float, 2>& z){
    //Tensor<float, 2> temp_max = z.colwise().maxCoeff();
    Tensor<float, 2> temp_max = z.reduce(dims_colwise, max_comparer);
    Tensor<float, 2> temp_exp{z.dimension(0), z.dimension(1)};

    for(int i{0}; i < z.dimension(1); i++){
        temp_exp.chip(i, 1) = z.chip(i, 1).unaryExpr(usrExp(temp_max(i)));
    }

    //Tensor<float, 1> temp_sum = temp_exp.colwise().sum().unaryExpr(std::ref(usrLog));
    Tensor<float, 1> temp_sum = temp_exp.sum(dims_colwise).unaryExpr(std::ref(usrLog));
    Tensor<float, 2> res(z.dimension(0), z.dimension(1));
    for(int i{0}; i < res.dimension(1); i++){
        //res.chip(i, 1) = z.chip(i, 1).unaryExpr(usrExp(temp_sum(i) + temp_max(i)));
        res.chip(i, 1) = z.chip(i, 1).unaryExpr(usrExp(temp_sum(i) + temp_max(i)));
    }
    return res;
}

Tensor<float, 2> SoftMaxLayer::grad_act(const Tensor<float, 2>& z){
    //Tensor<float, 2> temp_max = z.colwise().maxCoeff();
    Tensor<float, 2> temp_max = z.reduce(dims_colwise, max_comparer);
    Tensor<float, 2> temp_exp{z.dimension(0), z.dimension(1)};

    for(int i{0}; i < z.dimension(1); i++){
        temp_exp.chip(i, 1) = z.chip(i, 1).unaryExpr(usrExp(temp_max(i)));
    }

    //Tensor<float, 1> temp_sum = temp_exp.colwise().sum().unaryExpr(std::ref(usrLog));
    Tensor<float, 1> temp_sum = temp_exp.sum(dims_colwise).unaryExpr(std::ref(usrLog));
    Tensor<float, 2> res(z.dimension(0), z.dimension(1));
    for(int i{0}; i < res.dimension(1); i++){
        res.chip(i, 1) = z.chip(i, 1).unaryExpr(usrExp(temp_sum(i) + temp_max(i)));
    }
    return  res - res.cwiseProduct(res); 
}


