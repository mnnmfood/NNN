
#include "layers.h"
#include <Eigen/Dense>
#include <iostream>
#include "eigenFuns.h"
#include "typedefs.h"

inline const std::array<int, 1> dims_colwise {0};
inline const std::array<int, 1> dims_rowwise {1};

scalar_comparer_op<float> min_comparer([](float x, float y)->float {return x < y ? x: y;});
scalar_comparer_op<float> max_comparer([](float x, float y)->float {return x > y ? x: y;});

inline const Eigen::array<Eigen::IndexPair<int>, 1> product_dims = 
  {Eigen::IndexPair<int>(1, 0) };
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
BaseLayer::BaseLayer(size_t out_dims, size_t in_dims) 
:_out_num_dims{out_dims}, _in_num_dims {in_dims} {}
BaseLayer* BaseLayer::next(){return _next;}
BaseLayer* BaseLayer::prev(){return _prev;}

// Input Layer
InputLayer::InputLayer(size_t size)
    :Layer {size}
{}

Tensor<float> InputLayer::get_act(){
    return _act;
}

Tensor<float, 2> InputLayer::get_grad(){
    return _act;
}
void InputLayer::init(size_t n_samples){
    _act = Tensor<float, 2>(_size, n_samples);
}

void InputLayer::fwd(const Tensor<float, 2>& input){
    _act = input;
}



// Fully connected layer
FCLayer::FCLayer(size_t size)
   :Layer {size}
{
}

void FCLayer::initParams(){
    size_t prev_size = _prev->_size;
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
    _winputs = vecSum<float>(_weights.contract(_prev->get_act(), product_dims)
                            , _biases, false);
    _act = act(_winputs);
}

void FCLayer::bwd(const Tensor<float, 2>& cost_grad){
    assert(_next == nullptr);
    _nabla_b = cost_grad * grad_act(_winputs);
    _nabla_w = _nabla_b.contract(transposed(_prev->get_act()), product_dims);
    _grad = transposed(_weights).contract(_nabla_b, product_dims);
}

void FCLayer::bwd(){
    assert(_next != nullptr);
    _nabla_b = _next->get_grad() * grad_act(_winputs);
    _nabla_w = _nabla_b.contract(transposed(_prev->get_act()), product_dims);

    _grad = transposed(_weights).contract(_nabla_b, product_dims);
}

// TODO: updating method should be specific to optimization strategy,
// this should not be here
void FCLayer::update(float rate, float mu, float size){
    _weights = (1 - rate * mu / size) * _weights.eval()- (rate / size) * _nabla_w;
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
    Tensor<float, 2> temp_max = z.reduce(dims_colwise, max_comparer);
    Tensor<float, 2> temp_exp{z.dimension(0), z.dimension(1)};

    for(int i{0}; i < z.dimension(1); i++){
        temp_exp.chip(i, 1) = z.chip(i, 1).unaryExpr(usrExp(temp_max(i)));
    }

    Tensor<float, 1> temp_sum = temp_exp.sum(dims_colwise).unaryExpr(std::ref(usrLog));
    Tensor<float, 2> res(z.dimension(0), z.dimension(1));
    for(int i{0}; i < res.dimension(1); i++){
        res.chip(i, 1) = z.chip(i, 1).unaryExpr(usrExp(temp_sum(i) + temp_max(i)));
    }
    return res;
}

Tensor<float, 2> SoftMaxLayer::grad_act(const Tensor<float, 2>& z){
    Tensor<float, 2> temp_max = z.reduce(dims_colwise, max_comparer);
    Tensor<float, 2> temp_exp{z.dimension(0), z.dimension(1)};

    for(int i{0}; i < z.dimension(1); i++){
        temp_exp.chip(i, 1) = z.chip(i, 1).unaryExpr(usrExp(temp_max(i)));
    }

    Tensor<float, 1> temp_sum = temp_exp.sum(dims_colwise).unaryExpr(std::ref(usrLog));
    Tensor<float, 2> res(z.dimension(0), z.dimension(1));
    for(int i{0}; i < res.dimension(1); i++){
        res.chip(i, 1) = z.chip(i, 1).unaryExpr(usrExp(temp_sum(i) + temp_max(i)));
    }
    return  res - res * res; 
    //return  res - prod1(res, res); 
}

// Convolutional layer

#if 0
ConvolLayer::ConvolLayer(std::array<int, 3> shape)
    :Layer{shape[0]*shape[1]*shape[2]}, _shape{shape}
{
}

void ConvolLayer::init(size_t n_samples){
    _output_shape = {static_cast<int>(_size), static_cast<int>(n_samples)}; 
    _act = Tensor<float, 4>(_shape[0], _shape[1], _shape[2], n_samples);
    _grad = Tensor<float, 4>(_shape[0], _shape[1], _shape[2], n_samples);
    _winputs = Tensor<float, 4>(_shape[0], _shape[1], _shape[2], n_samples);
    _nabla_b = Tensor<float, 4>(_shape[0], _shape[1], _shape[2], n_samples);
    _nabla_w = Tensor<float, 4>(_shape[0], _shape[1], _shape[2], n_samples);
}

void ConvolLayer::initParams(){
    size_t prev_size = _prev->getSize();
    NormalSample sampleFun(0.0f, 1.0f / std::sqrt(
        static_cast<float>(prev_size)
    ));
    _weights = Tensor<float, 3>(_shape[0], _shape[1], _shape[2]);
    _biases = Tensor<float, 2>(_shape[2]);
}

Tensor<float, 4> ConvolLayer::get_act() {return _act;}
Tensor<float, 4> ConvolLayer::get_grad() {return _grad;}

void ConvolLayer::fwd(){
    assert(_prev != nullptr);
    _winputs = _prev->get_act().reshape()
}
#endif