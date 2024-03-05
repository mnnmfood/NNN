
#include <iostream>
#include "typedefs.h"
#include "layers.h"
#include "utils.h"
#include "convolutions.h"
#include "max_poling.h"

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
BaseLayer::BaseLayer(size_t out_dims, size_t in_dims, std::string_view d)
    :_out_num_dims{ out_dims }, _in_num_dims{ in_dims }, _descriptor{d}
{}
BaseLayer* BaseLayer::next(){return _next;}
BaseLayer* BaseLayer::prev(){return _prev;}
std::string BaseLayer::which() { return _descriptor; }
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

void FCLayer::fwd(ThreadPoolDevice* device){
    assert(_prev != nullptr);
    _winputs = vecSum(_weights.contract(prev_act(), 
        product_dims), _biases, false);
    Tensor<float, 2> temp = act(_winputs);
    _act = act(_winputs);
}

void FCLayer::fwd(TensorWrapper<float>&&, ThreadPoolDevice* device){}
void FCLayer::bwd(TensorWrapper<float>&& cost_grad, ThreadPoolDevice* device){
    assert(_next == nullptr);
    _nabla_b = cost_grad.get(_out_batch_shape) * grad_act(_winputs);
    _nabla_w = _nabla_b.contract(transposed(prev_act()), product_dims);
    _grad = transposed(_weights).contract(_nabla_b, product_dims);
}

void FCLayer::bwd(ThreadPoolDevice* device){
    assert(_next != nullptr);
    _nabla_b = next_grad() * grad_act(_winputs);
    Tensor<float, 2> temp = transposed(prev_act());
    Tensor<float, 2> temp2 = _nabla_b.contract(
        temp, product_dims
    );
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
    Tensor<float, 1> temp_max = z.reduce(dims_colwise, max_comparer);
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
    Tensor<float, 1> temp_max = z.reduce(dims_colwise, max_comparer);
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
}

// Convolutional layer

ConvolLayer::ConvolLayer(std::array<Index, 3> shape)
    :Layer{}, _shape{shape}
{
    std::array<Index, 4> channels_shape;
    // [depth, channels, rows, columns]
    channels_shape[0] = shape[0];
    channels_shape[1] = 1;
    channels_shape[2] = shape[1];
    channels_shape[3] = shape[2];
    NormalSample sampleFun(0.0f, 1.0f / std::sqrt(
        static_cast<float>(shape[1] * shape[2])
    ));
    _weights = weight_t(channels_shape).unaryExpr(std::ref(sampleFun));
    _nabla_w = nabla_weight_t(channels_shape);
    _biases = bias_t(shape[0]).unaryExpr(std::ref(sampleFun));
}

void ConvolLayer::initParams(){
    _in_shape = prev_shape();
    _out_shape = {
        1,
        _in_shape[1] - _weights.dimension(2) + 1,
        _in_shape[2] - _weights.dimension(3) + 1,
        _in_shape[3] * _weights.dimension(0)
    }; 
}

void ConvolLayer::init(Index batch_size){
    std::copy(_in_shape.begin(), _in_shape.end(), 
        _in_batch_shape.begin());
    std::copy(_out_shape.begin(), _out_shape.end(), 
        _out_batch_shape.begin());

    _out_batch_shape.back() = batch_size;
    _in_batch_shape.back() = batch_size;

    _act = out_t(_out_batch_shape);
    _grad = in_t(_in_batch_shape);

    std::array<Index, 2> shape_nabla_b{
        _shape[0], batch_size
    };
    _nabla_b = nabla_b_t(shape_nabla_b); 
    _nabla_b.setConstant(0);
}

void ConvolLayer::fwd(ThreadPoolDevice* device){
    _act.device(*device) = convolveBatch(prev_act(), _weights);
    //imwrite(_act.chip(0, 4).chip(0, 3).chip(0, 0), "./" + std::to_string(_i) + "_conv.png");
    //imwrite(prev_act().chip(0, 4).chip(0, 3).chip(0, 0), "./" + std::to_string(_i) + "_orig.png");
    //_i++;
}

void ConvolLayer::fwd(TensorWrapper<float>&&, ThreadPoolDevice* device){}
void ConvolLayer::bwd(TensorWrapper<float>&&, ThreadPoolDevice* device){}

void ConvolLayer::bwd(ThreadPoolDevice* device) {

    Index batch_size {_out_batch_shape.back()};
    Index depth {_shape[0]};
    Index in_depth {_in_shape[3]};
    Index im_rows{ _in_shape[1] };
    Index im_cols{ _in_shape[2] };
    Index ker_rows{ _shape[1] };
    Index ker_cols{ _shape[2] };

    Eigen::DSizes<Index, 5> offsets_output{ 0, 0, 0, 0, 0 };
    Eigen::DSizes<Index, 5> extents_output{
        _out_batch_shape[0],
        _out_batch_shape[1],
        _out_batch_shape[2],
        depth,
        _out_batch_shape[4],
    };

    for (Index k{ 0 }; k < in_depth; k++) {
        offsets_output[3] = k * depth;
        _grad.chip(k, 3).device(*device) = backwardsConvolveInput(
            next_grad().slice(offsets_output, extents_output),
            _weights, im_rows, im_cols);
        _nabla_w.device(*device) += backwardsConvolveKernel(
            prev_act().chip(k, 3),
            next_grad().slice(offsets_output, extents_output),
            ker_rows, ker_cols);
    }
}

PoolingLayer::PoolingLayer(std::array<Index, 2> shape, Index stride)
    :Layer{}, _shape{shape}, _stride{stride}
{
}

void PoolingLayer::initParams(){
    _in_shape = prev_shape();
    _out_shape = {
        _in_shape[0],
        static_cast<Index>(ceil( 
            (static_cast<float>(_in_shape[1] - _shape[0] + 1)) / _stride)),
        static_cast<Index>(ceil( 
            (static_cast<float>(_in_shape[2] - _shape[1] + 1)) / _stride)),
        _in_shape[3]
    };
}

void PoolingLayer::init(Index batch_size){
    std::copy(_in_shape.begin(), _in_shape.end(), 
        _in_batch_shape.begin());
    std::copy(_out_shape.begin(), _out_shape.end(), 
        _out_batch_shape.begin());
    _out_batch_shape.back() = batch_size;
    _in_batch_shape.back() = batch_size;
    _grad = in_t(_in_batch_shape);
    _act = out_t(_out_batch_shape);
    _argmax = Tensor<Index, 5>(_out_batch_shape);
}
void PoolingLayer::fwd(TensorWrapper<float>&&, ThreadPoolDevice* device){}
void PoolingLayer::bwd(TensorWrapper<float>&&, ThreadPoolDevice* device){}


void PoolingLayer::fwd(ThreadPoolDevice* device) {
    const Index ir = _in_shape[1];
    const Index ic = _in_shape[2];
    const Index kr = _shape[0];
    const Index kc = _shape[1];
    const Index stride = _stride;
    const Index depth = _in_shape[3];
    const Index batch = _in_batch_shape[4];

    max_pooling(prev_act(), ir, ic, depth, batch, kr, kc, stride, _act, _argmax);

    //imwrite(_act.chip(0, 4).chip(0, 3).chip(0, 0), "./" + std::to_string(_i) + "_max_pool.png");
    //imwrite(prev_act().chip(0, 4).chip(0, 3).chip(0, 0), "./" + std::to_string(_i) + "_conv_orig.png");
    //_i++;
}

void PoolingLayer::bwd(ThreadPoolDevice* device) {
    const Index outr = _out_shape[1];
    const Index outc = _out_shape[2];
    const Index depth = _out_shape[3];
    const Index batch= _out_batch_shape[4];
    
    Tensor<float, 5> grad = next_grad();
    //_grad.setConstant(0.0f);
    Index idx_flat = 0;
    for(Index i{0}; i < batch; i++){ // batch
        for(Index k{0}; k < depth; k++){ // depth
            for(Index r{0}; r < outr; r++){ // rows
                for(Index c{0}; c < outc; c++){ // cols
                    idx_flat = _argmax(0, r, c, k, i);
                    _grad(idx_flat) = grad(0, r, c, k, i);
                }
            }
        }
    }
}
