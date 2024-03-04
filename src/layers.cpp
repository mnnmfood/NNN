
#include <iostream>
#include "typedefs.h"
#include "layers.h"
#include "utils.h"
#include "convolutions.h"

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
    Tensor<float, 5> temp = prev_act();
    //_act = convolveBatch(prev_act(), _weights);
    //_act = convolveBatch(temp, _weights);
    _act.device(*device) = convolveBatch(temp, _weights);
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
    auto grad = next_grad();
    auto act = prev_act();

    for (Index k{ 0 }; k < in_depth; k++) {
        offsets_output[3] = k * depth;
        //_grad.chip(k, 3) = backwardsConvolveInput(
        _grad.chip(k, 3).device(*device) = backwardsConvolveInput(
            grad.slice(offsets_output, extents_output),
            _weights, im_rows, im_cols);
        //_nabla_w += backwardsConvolveKernel(
        _nabla_w.device(*device) += backwardsConvolveKernel(
            act.chip(k, 3),
            grad.slice(offsets_output, extents_output),
            ker_rows, ker_cols);
    }
}

#if 1
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
}
void PoolingLayer::fwd(TensorWrapper<float>&&, ThreadPoolDevice* device){}
void PoolingLayer::bwd(TensorWrapper<float>&&, ThreadPoolDevice* device){}

void PoolingLayer::fwd(ThreadPoolDevice* device){
    //auto act = prev_act();
    Tensor<float, 6> patches;    
    patches = prev_act().extract_image_patches(
        _shape[0], _shape[1], _stride, _stride, 1, 1, Eigen::PADDING_VALID
    );

    Tensor<Index, 5> maxEachRow;
    maxEachRow = patches.abs().argmax(1);
    const std::array<Index, 1> row_max_dims ({1});
    _maxCol = patches.abs().maximum(row_max_dims).argmax(1);

    _maxRow = Tensor<Index, 4>(_maxCol.dimensions());
    
    // [channels, row, col, patch, in_depth, batch]
    // [0, row, col, patch, batch]
    for(Index batch{0}; batch < patches.dimension(5); batch++){
        for(Index in_depth{0}; in_depth < patches.dimension(4); in_depth++){
            for(Index patch{0}; patch < patches.dimension(3); patch++){
                //_maxRow(k, l, i) = maxEachRow(k, _maxCol(k, l, i), l, i);
                _maxRow(0, patch, in_depth, batch) = 
                    maxEachRow(0, _maxCol(0, patch, in_depth, batch), patch, in_depth, batch);
            }
        }
    }

    Index row_l{0}, col_l{0};
    Index cols {_act.dimension(2)};
    Index max_row_patch, max_col_patch;
    for(Index batch{0}; batch < patches.dimension(5); batch++){
        for(Index in_depth{0}; in_depth < patches.dimension(4); in_depth++){
            for(Index patch{0}; patch < patches.dimension(3); patch++){
                row_l = patch % cols; 
                col_l = patch / cols;
                max_col_patch = _maxCol(0, patch, in_depth, batch);
                max_row_patch = _maxRow(0, patch, in_depth, batch);
                _act(0, row_l, col_l, in_depth, batch) = 
                    patches(0, max_row_patch, max_col_patch, patch, in_depth, batch);
                //_act(k, row_l, col_l, i) = 
                //    patches(k, max_row_patch, max_col_patch, l, i);
            }
        }
    }
}

void PoolingLayer::bwd(ThreadPoolDevice* device){
#if 1
    Tensor<float, 5> grad = next_grad();
    Tensor<float, 5> act = prev_act().abs();
    _grad.setConstant(0);

    Index idx;
    Index rows_l {grad.dimension(2)};
    Index max_row_patch, max_col_patch;
    for(Index batch{0}; batch < grad.dimension(4); batch++){ // i
        for(Index out_depth{0}; out_depth < grad.dimension(3); out_depth++){ // k
            for(Index row{0}; row < grad.dimension(1); row++){ // l
                for(Index col{0}; col < grad.dimension(2); col++){ // m
                    idx = col*rows_l + row;
                    max_row_patch = _maxRow(0, idx, out_depth, batch) + row * _stride;
                    max_col_patch = _maxCol(0, idx, out_depth, batch) + col * _stride;
                    //_grad(k, max_row_patch,  max_col_patch, i) = grad(k, l, m, i);
                    _grad(0, max_row_patch,  max_col_patch, out_depth, batch) = 
                        grad(0, row, col, out_depth, batch);
                }
            }
        }
    }
#endif
}
#endif