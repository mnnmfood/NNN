
#include "layers.h"
#include <Eigen/Dense>
#include <iostream>
#include "eigenFuns.h"
#include "typedefs.h"

inline const std::array<int, 1> dims_colwise {0};
inline const std::array<int, 1> dims_rowwise {1};
inline const std::array<int, 2> dims_convolve {0, 1};

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
BaseLayer::BaseLayer(int depth) :_depth{depth} {}
BaseLayer* BaseLayer::next(){return _next;}
BaseLayer* BaseLayer::prev(){return _prev;}

// Fully connected layer
FCLayer::FCLayer(Index size)
   :Layer {std::array<Index, 1> {size}}
{
    _output_shape = _shape;
}

void FCLayer::initParams(){
    // Make sure previous Layer is compatible
    assert(_prev->_depth == 1);
    std::array<Index, 1> prev_size = dynamic_cast<Layer<1>*>(_prev)->_shape;
    NormalSample sampleFun(0.0f, 1.0f / std::sqrt(
        static_cast<float>(prev_size[0])
    ));

    _weights = Tensor<float, 2>(_shape[0], prev_size[0]).unaryExpr(std::ref(sampleFun));
    _biases = Tensor<float, 1>(_shape[0]).unaryExpr(std::ref(sampleFun));
}

void FCLayer::init(Index n_samples){
    _act = Tensor<float, 2>(_shape[0], n_samples); 
    _grad = Tensor<float, 2>(_shape[0], n_samples); 
    _winputs = Tensor<float, 2>(_shape[0], n_samples);
    _nabla_b = Tensor<float, 2>(_shape[0], n_samples); 
    _nabla_w = Tensor<float, 2>(_shape[0], n_samples); 
}

Tensor<float, 2> FCLayer::get_act(){return _act;}
Tensor<float, 2> FCLayer::get_grad(){return _grad;}

void FCLayer::fwd(){
    assert(_prev != nullptr);
    Layer<1>* prev {dynamic_cast<Layer<1>*>(_prev)};
    _winputs = vecSum<float>(_weights.contract(prev->get_act(), product_dims)
                            , _biases, false);
    _act = act(_winputs);
}

void FCLayer::fwd(const Tensor<float, 2>& input){}

void FCLayer::bwd(const Tensor<float, 2>& cost_grad){
    assert(_next == nullptr);
    Layer<1>* prev {dynamic_cast<Layer<1>*>(_prev)};
    _nabla_b = cost_grad * grad_act(_winputs);
    _nabla_w = _nabla_b.contract(transposed(prev->get_act()), product_dims);
    _grad = transposed(_weights).contract(_nabla_b, product_dims);
}

void FCLayer::bwd(){
    assert(_next != nullptr);
    Layer<1>* next {dynamic_cast<Layer<1>*>(_next)};
    Layer<1>* prev {dynamic_cast<Layer<1>*>(_prev)};
    _nabla_b = next->get_grad() * grad_act(_winputs);
    _nabla_w = _nabla_b.contract(transposed(prev->get_act()), product_dims);

    _grad = transposed(_weights).contract(_nabla_b, product_dims);
}

// TODO: updating method should be specific to optimization strategy,
// this should not be here
void FCLayer::update(float rate, float mu, float size){
    _weights = (1 - rate * mu / size) * _weights.eval()- (rate / size) * _nabla_w;
    _biases -= (rate / size) * (_nabla_b.sum(dims_rowwise));
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
TanhLayer::TanhLayer(Index size) :FCLayer{size}{}
Tensor<float, 2> TanhLayer::act(const Tensor<float, 2>& z){
    return z.unaryExpr(std::ref(tanhc));
}

Tensor<float, 2> TanhLayer::grad_act(const Tensor<float, 2>& z){
    return z.unaryExpr(std::ref(tanh_prime));
}

// SoftMax Layer
SoftMaxLayer::SoftMaxLayer(Index size) :FCLayer{size}{}
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
    :Layer<3> {shape}
{
}

void ConvolLayer::init(Index n_samples){
    std::array<Index, 3> prev_shape = 
        dynamic_cast<Layer<3>*>(_prev)->_output_shape;
    _act = Tensor<float, 4>(_output_shape[0], _output_shape[1], 
        _output_shape[2], n_samples);

    _grad = Tensor<float, 4>(prev_shape[0], prev_shape[1], 
        prev_shape[2], n_samples);
    _winputs = Tensor<float, 4>(prev_shape[0], prev_shape[1], 
        prev_shape[2], n_samples);

    _nabla_b = Tensor<float, 2>(_shape[2], n_samples);
    _nabla_w = Tensor<float, 4>(_shape[0], _shape[1], 
        _shape[2], n_samples);
}

void ConvolLayer::initParams(){
    // Make sure previous Layer is compatible
    assert(_prev->_depth == 3);
    NormalSample sampleFun(0.0f, 1.0f / std::sqrt(
        static_cast<float>(_shape[0]*_shape[1])
        ));
    std::array<Index, 3> prev_shape = 
        dynamic_cast<Layer<3>*>(_prev)->_output_shape;
    _output_shape = std::array<Index, 3>{
        prev_shape[0] - _shape[0] +1,
        prev_shape[1] - _shape[1] +1,
        prev_shape[2] * _shape[2]};

    _weights = Tensor<float, 3>(_shape)
        .unaryExpr(std::ref(sampleFun));
    _biases = Tensor<float, 1>(_shape[2])
        .unaryExpr(std::ref(sampleFun));
}

Tensor<float, 4> ConvolLayer::get_act() {return _act;}
Tensor<float, 4> ConvolLayer::get_grad() {return _grad;}

void ConvolLayer::fwd(){
    assert(_prev != nullptr);
    Layer<1>* prev {dynamic_cast<Layer<1>*>(_prev)};
    _winputs = prev->get_act();
}

void ConvolLayer::bwd(){
    assert(_next != nullptr);
    Layer<1>* next {dynamic_cast<Layer<1>*>(_next)};
    Layer<1>* prev {dynamic_cast<Layer<1>*>(_prev)};
}

Tensor<float, 4> ConvolLayer::act(const Tensor<float, 4>& z){return z;}
Tensor<float, 4> ConvolLayer::grad_act(const Tensor<float, 4>& z){return z;}

void ConvolLayer::update(float rate, float mu, float size){
    _weights = (1 - rate * mu / size) * _weights.eval()- (rate / size) * _nabla_w;
    _biases -= (rate / size) * (_nabla_b.sum(dims_rowwise));
}

void ConvolLayer::fwd(const Tensor<float, 4>&){};
void ConvolLayer::bwd(const Tensor<float, 4>&){};
