
#include "layers.h"
#include <Eigen/Dense>
#include <iostream>

using Eigen::MatrixXf;
using Eigen::Dynamic;
using Eigen::VectorXf;

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
    _act = MatrixXf(_size, n_samples);
}

void InputLayer::fwd(const MatrixXf& input){
    _act = input;
}

MatrixXf InputLayer::get_act(){
    return _act;
}

MatrixXf InputLayer::get_grad(){
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

    _weights = MatrixXf(_size, prev_size).unaryExpr(std::ref(sampleFun));
    _biases = VectorXf(_size).unaryExpr(std::ref(sampleFun));
}

void FCLayer::init(size_t n_samples){
    _act = MatrixXf(_size, n_samples); 
    _grad = MatrixXf(_size, n_samples); 
    _winputs = MatrixXf(_size, n_samples);
    _nabla_b = MatrixXf(_size, n_samples); 
    _nabla_w = MatrixXf(_size, n_samples); 
}

MatrixXf FCLayer::get_act(){return _act;}
MatrixXf FCLayer::get_grad(){return _grad;}

void FCLayer::fwd(){
    assert(_prev != nullptr);
    _winputs = (_weights * _prev->get_act()).colwise() + _biases ;
    _act = act(_winputs);
}

void FCLayer::bwd(const MatrixXf& cost_grad){
    assert(_next == nullptr);
    _nabla_b = cost_grad.cwiseProduct(grad_act(_winputs));
    _nabla_w = _nabla_b * _prev->get_act().transpose();
    _grad = _weights.transpose() * _nabla_b;
}

void FCLayer::bwd(){
    assert(_next != nullptr);
    _nabla_b = _next->get_grad().cwiseProduct(grad_act(_winputs));
    _nabla_w = _nabla_b * _prev->get_act().transpose();
    _grad = _weights.transpose() * _nabla_b;
}

// TODO: updating method should be specific to optimization strategy,
// this should not be here
void FCLayer::update(float rate, float mu, float size){
    _weights = (1 - rate * mu / size) * _weights.eval()- (rate / size) * _nabla_w;
    _biases -= (rate / size) * (_nabla_b.rowwise().sum());
}

// Sigmoid layer
SigmoidLayer::SigmoidLayer(size_t size) :FCLayer{size}{}

MatrixXf SigmoidLayer::act(const MatrixXf& z){
    return z.unaryExpr(std::ref(logistic));
}

MatrixXf SigmoidLayer::grad_act(const MatrixXf& z){
    return z.unaryExpr(std::ref(logistic_prime));
}

// Tanh Layer
MatrixXf TanhLayer::act(const MatrixXf& z){
    return z.unaryExpr(std::ref(tanhc));
}

MatrixXf TanhLayer::grad_act(const MatrixXf& z){
    return z.unaryExpr(std::ref(tanh_prime));
}

// SoftMax Layer
MatrixXf SoftMaxLayer::act(const MatrixXf& z){
    MatrixXf temp_max = z.colwise().maxCoeff();
    MatrixXf temp_exp{z.rows(), z.cols()};

    for(int i{0}; i < z.cols(); i++){
        temp_exp.col(i) = z.col(i).unaryExpr(usrExp(temp_max(i)));
    }

    VectorXf temp_sum = temp_exp.colwise().sum().unaryExpr(std::ref(usrLog));
    MatrixXf res(z.rows(), z.cols());
    for(int i{0}; i < res.cols(); i++){
        res.col(i) = z.col(i).unaryExpr(usrExp(temp_sum(i) + temp_max(i)));
    }
    return res;
}

MatrixXf SoftMaxLayer::grad_act(const MatrixXf& z){
    MatrixXf temp_max = z.colwise().maxCoeff();
    MatrixXf temp_exp{z.rows(), z.cols()};

    for(int i{0}; i < z.cols(); i++){
        temp_exp.col(i) = z.col(i).unaryExpr(usrExp(temp_max(i)));
    }

    VectorXf temp_sum = temp_exp.colwise().sum().unaryExpr(std::ref(usrLog));
    MatrixXf res(z.rows(), z.cols());
    for(int i{0}; i < res.cols(); i++){
        res.col(i) = z.col(i).unaryExpr(usrExp(temp_sum(i) + temp_max(i)));
    }
    return  res - res.cwiseProduct(res); 
}


