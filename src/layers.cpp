
#include "layers.h"
#include <Eigen/Dense>
#include <iostream>

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

// Fully connected layer
FCLayer::FCLayer(Index size) 
    :Layer{std::array<Index, 1>{size}, std::array<Index, 1>{size}},
     _shape{size}{ 
    std::copy(_shape.begin(), _shape.end(), batch_shape.begin());
}

void FCLayer::initParams(){
    in_shape_t prev_size = _prev->out_shape().get<in_shape_t>();
    NormalSample sampleFun(0.0f, 1.0f / std::sqrt(
        static_cast<float>(_shape[0])
    ));
    std::array<Index, 2> temp {_shape[0], prev_size[0]};
    _weights = weight_t(temp).unaryExpr(std::ref(sampleFun));
    _biases = bias_t(_shape[0]).unaryExpr(std::ref(sampleFun));
}

void FCLayer::init(Index batch_size){
    batch_shape.back() = batch_size;
    _act = out_t(batch_shape); 
    _grad = in_t(batch_shape); 
    _winputs = in_t(batch_shape);
    _nabla_b = nabla_b_t(batch_shape); 
    _nabla_w = nabla_weight_t(batch_shape); 
}

void FCLayer::fwd(){
    assert(_prev != nullptr);
    std::cout << _in_shape[0] << ", " << _in_shape[1];
    _winputs = vecSum(_weights.contract(_prev->get_act().get(_in_shape), 
        product_dims), _biases, false);
    _act = act(_winputs);
}

void FCLayer::fwd(TensorWrapper<float>&&){}
void FCLayer::bwd(TensorWrapper<float>&& cost_grad){
    assert(_next == nullptr);
    _nabla_b = cost_grad.get(_out_shape) * grad_act(_winputs);
    _nabla_w = _nabla_b.contract(transposed(_prev->get_act().get(_in_shape)), product_dims);
    _grad = transposed(_weights).contract(_nabla_b, product_dims);
}

void FCLayer::bwd(){
    assert(_next != nullptr);
    _nabla_b = _next->get_grad().get(_out_shape) * grad_act(_winputs);
    _nabla_w = _nabla_b.contract(
        transposed(_prev->get_act().get(_in_shape)), product_dims);

    _grad = transposed(_weights).contract(_nabla_b, product_dims);
}


// Sigmoid layer
SigmoidLayer::SigmoidLayer(Index size) :FCLayer{size}{}

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