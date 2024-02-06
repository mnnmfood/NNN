
#include "layers.h"
#include <Eigen/Dense>
#include <iostream>

scalar_comparer_op<float> min_comparer([](float x, float y)->float {return x < y ? x: y;});
scalar_comparer_op<float> max_comparer([](float x, float y)->float {return x > y ? x: y;});

inline const Eigen::array<Eigen::IndexPair<int>, 1> product_dims = 
  {Eigen::IndexPair<int>(1, 0) };

inline const Eigen::array<bool, 4> reverse_dims {true, true, 
                                                false, false};
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
:_out_num_dims{out_dims}, _in_num_dims {in_dims} {
}
BaseLayer* BaseLayer::next(){return _next;}
BaseLayer* BaseLayer::prev(){return _prev;}

// Fully connected layer
FCLayer::FCLayer(Index size) 
    :Layer{std::array<Index, 1>{size}}, _shape{size}
{ 
    std::copy(_shape.begin(), _shape.end(), _out_batch_shape.begin());
}

void FCLayer::initParams(){
    _in_shape = prev_shape();
    std::copy(_in_shape.begin(), _in_shape.end(), _in_batch_shape.begin());

    std::array<Index, 2> temp {_out_shape[0], _in_shape[0]};
    NormalSample sampleFun(0.0f, 1.0f / std::sqrt(
        static_cast<float>(temp[0] * temp[1])
    ));
    _weights = weight_t(temp).unaryExpr(std::ref(sampleFun));
    _biases = bias_t(_shape[0]).unaryExpr(std::ref(sampleFun));
}

void FCLayer::init(Index batch_size){
    _out_batch_shape.back() = batch_size;
    _in_batch_shape.back() = batch_size;
    _act = out_t(_out_batch_shape); 
    _grad = in_t(_in_batch_shape); 
    _winputs = in_t(_in_batch_shape);
    _nabla_b = nabla_b_t(_out_batch_shape); 
    _nabla_w = nabla_weight_t(_out_batch_shape); 
}

void FCLayer::fwd(){
    assert(_prev != nullptr);
    _winputs = vecSum(_weights.contract(prev_act(), 
        product_dims), _biases, false);
    _act = act(_winputs);
}

void FCLayer::fwd(TensorWrapper<float>&&){}
void FCLayer::bwd(TensorWrapper<float>&& cost_grad){
    assert(_next == nullptr);
    _nabla_b = cost_grad.get(_out_batch_shape) * grad_act(_winputs);
    _nabla_w = _nabla_b.contract(transposed(prev_act()), product_dims);
    _grad = transposed(_weights).contract(_nabla_b, product_dims);
}

void FCLayer::bwd(){
    assert(_next != nullptr);
    _nabla_b = next_grad() * grad_act(_winputs);
    _nabla_w = _nabla_b.contract(
        transposed(prev_act()), product_dims);

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

ConvolLayer::ConvolLayer(std::array<Index, 3> shape)
    :Layer{}, _shape{shape}
{
    NormalSample sampleFun(0.0f, 1.0f / std::sqrt(
        static_cast<float>(shape[0] * shape[1])
    ));
    _weights = weight_t(shape).unaryExpr(std::ref(sampleFun));
    _nabla_w = nabla_weight_t(shape);
    _biases = bias_t(shape[2]).unaryExpr(std::ref(sampleFun));
}

void ConvolLayer::init(Index batch_size){
    _in_shape = prev_shape();
    _out_shape = {_in_shape[0] - _weights.dimension(0) + 1,
        _in_shape[1] - _weights.dimension(1) + 1, 
        _in_shape[2] * _weights.dimension(2)};

    std::copy(_in_shape.begin(), _in_shape.end(), 
        _in_batch_shape.begin());
    std::copy(_out_shape.begin(), _out_shape.end(), 
        _out_batch_shape.begin());

    _out_batch_shape.back() = batch_size;
    _in_batch_shape.back() = batch_size;
    _grad = in_t(_in_batch_shape);
}

void ConvolLayer::initParams(){
}

void ConvolLayer::fwd(){
    _act = convolveBatch(prev_act(), _weights, valid);
}

void ConvolLayer::bwd(){
    gradWeights();
    gradLoss();
}

void ConvolLayer::fwd(TensorWrapper<float>&&){}
void ConvolLayer::bwd(TensorWrapper<float>&&){}


inline const std::array<Index, 1> sum_dims {2};
void ConvolLayer::gradWeights(){
    Index batch_size {_out_batch_shape.back()};
    Index depth {_shape.back()};
    Index in_depth {_shape.back()};
    Tensor<float, 4> act = prev_act();
    Tensor<float, 4> grad = next_grad();

    std::array<Index, 3> offsets;
    std::array<Index, 3> extents{
        grad.dimension(0),
        grad.dimension(1),
        in_depth
    };

    for(Index i{0}; i < batch_size; i++){
        for(Index k{0}; k < depth; k++){
            offsets = {0, 0, in_depth*k};
            _nabla_w.chip(k, 2) += convolveEach(
                act.chip(i, 3),
                grad.chip(i, 3)
                .slice(offsets, extents)
            ).sum(sum_dims);
        }
    }
}

inline const std::array<bool, 3> flip_dims {true, true};
void ConvolLayer::gradLoss(){
    Index batch_size {_out_batch_shape.back()};
    Index depth {_shape.back()};
    Index in_depth {_shape.back()};
    Tensor<float, 4> grad  = next_grad();

    int kr {static_cast<int>(grad.dimension(0))};
    int kc {static_cast<int>(grad.dimension(1))};
    Eigen::array<std::pair<int, int>, 3> paddings;
    paddings = {std::make_pair(kr-1, kr-1), 
                std::make_pair(kc-1, kc-1),
                std::make_pair(0, 0)};
    Tensor<float, 3> rot_w = 
        _weights.reverse(flip_dims)
        .pad(paddings);

    std::array<Index, 3> offsets;
    std::array<Index, 3> extents{
        grad.dimension(0),
        grad.dimension(1),
        in_depth
    };
    for(Index i{0}; i < batch_size; i++){
        for(Index k{0}; k < depth; k++){
            offsets = {0, 0, in_depth*k};
            _grad.chip(i, 3) +=
                convolveKernels(
                    _weights.chip(k, 2),
                    grad.chip(i, 3)
                    .slice(offsets, extents)
                );
        }
    }
}