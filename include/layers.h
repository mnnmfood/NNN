#ifndef LAYERS_H
#define LAYERS_H

#include<Eigen/Dense>
#include<random>
#include "typedefs.h"
#include <Tensor.h>

inline const std::array<int, 1> dims_colwise {0};
inline const std::array<int, 1> dims_rowwise {1};

inline std::random_device rd{};
inline std::mt19937 gen{rd()};

class BaseLayer
{
public:
    size_t _out_num_dims;
    size_t _in_num_dims;
    BaseLayer* _next = nullptr;
    BaseLayer* _prev = nullptr;

    BaseLayer* next();
    BaseLayer* prev();
    BaseLayer(size_t, size_t);

    virtual void fwd() = 0;
    virtual void fwd(const TensorWrapper<float>&) = 0;
    virtual void bwd() = 0;
    virtual void bwd(const TensorWrapper<float>&) = 0;

    virtual void init(size_t){}
    virtual void initParams() = 0;

    virtual TensorWrapper<float> get_act() = 0;
    virtual TensorWrapper<float> get_grad() = 0;
    virtual TensorShape in_shape() = 0; 
    virtual TensorShape out_shape() = 0; 

    virtual ~BaseLayer() = default;

    virtual void update(float rate, float mu, float size) = 0; 
};

template<class Derived>
class Layer: public BaseLayer
{
protected:
    typedef typename traits<Derived>::out_shape_t out_shape_t;
    typedef typename traits<Derived>::in_shape_t in_shape_t;
    const size_t out_dims {std::tuple_size<out_shape_t>{}};
    const size_t in_dims {std::tuple_size<in_shape_t>{}};
    const size_t num_dims {traits<Derived>::NumDimensions};
    using out_t = Tensor<float, std::tuple_size<out_shape_t>{}+1>
    using in_t = Tensor<float, std::tuple_size<in_shape_t>{}+1>
    using weight_t = std::conditional<traits<Derived>::trainable, 
     Tensor<float, traits<Derived>::NumDimensions>, dummyTensor>::type;
    using bias_t = std::conditional<traits<Derived>::trainable, 
     Tensor<float, 1>, dummyTensor>::type;
    using nabla_weight_t = std::conditional<traits<Derived>::trainable, 
     Tensor<float, traits<Derived>::NumDimensions+1>, dummyTensor>::type;
    using nabla_b_t = std::conditional<traits<Derived>::trainable, 
     Tensor<float, 2>, dummyTensor>::type;

    bool _trainable = traits<Derived>::trainable;
    out_t _act;
    in_t _grad;
    in_t _winputs;
    weight_t _weights;
    bias_t _biases;
    nabla_weight_t _nabla_w;
    nabla_b_t _nabla_b;
    out_shape_t _out_shape;
    in_shape_t _in_shape;

public:
    Layer() :BaseLayer {out_dims, in_dims} {}
    TensorWrapper<float>& get_act(){
        return TensorWrapepr(_act);
    }
    TensorWrapper<float>& get_grad(){
        return TensorWrapper(_grad);
    }
    // TODO: updating method should be specific to optimization strategy,
    // this should not be here
    void update(float rate, float mu, float size){
        if(_trainable){
        _weights = (1 - rate * mu / size) * _weights.eval()- (rate / size) * _nabla_w;
        _biases -= (rate / size) * (_nabla_b.sum(dims_rowwise));
        }
    }
    TensorShape in_shape(){
        return TensorShape(_in_shape);
    }
    TensorShape out_shape(){
        return TensorShape(_out_shape);
    }
};

template<typename N>
class InputLayer: public Layer<InputLayer<N>>
{
    std::array<Index, N> _shape;
    std::array<Index, N+1> batch_shape;
public:
    const size_t _size = 0;
    InputLayer(std::array<Index, N> shape) :_shape{shape}{
        std::copy(_shape.begin(), _shape.end(), batch_shape);
    }
    void init(size_t n_samples){
        batch_shape.back() = n_samples;
        _act = Tensor<float, N>(batch_shape);
    }
    void initParams(){}
    void fwd(){}
    void fwd(const TensorWrapper<float>& input){
        _act = input.get(batch_size);
    }
    void bwd(){};
    void bwd(const TensorWrapper<float>&){};
};

class FCLayer: public Layer<FCLayer>
{
    std::array<Index, 1> _shape;
    std::array<Index, 2> batch_shape;
public:
    FCLayer(Index size);
    void init(Index n_samples);
    void initParams();

    void fwd();
    void bwd();

    void bwd(const Tensor<float, 2>&);

    virtual Tensor<float, 2> act(const Tensor<float, 2>&) = 0;
    virtual Tensor<float, 2> grad_act(const Tensor<float, 2>&) = 0;

    virtual ~FCLayer() = default;
};

class SigmoidLayer: public FCLayer
{
public:
    SigmoidLayer(size_t size); 
    Tensor<float, 2> act(const Tensor<float, 2>&);
    Tensor<float, 2> grad_act(const Tensor<float, 2>&);
};

class TanhLayer: public FCLayer
{
public:
    Tensor<float, 2> act(const Tensor<float, 2>&);
    Tensor<float, 2> grad_act(const Tensor<float, 2>&);
};

class SoftMaxLayer: public FCLayer
{
public:
    Tensor<float, 2> act(const Tensor<float, 2>&);
    Tensor<float, 2> grad_act(const Tensor<float, 2>&);
};

#if 0
class ConvolLayer:public Layer
{
protected:
    Tensor<float, 3> _weights;
    Tensor<float, 2> _biases;

    Tensor<float, 2> _winputs; 

    Tensor<float, 4> _act;
    Tensor<float, 4> _grad;

    Tensor<float, 2> _nabla_b;
    Tensor<float, 3> _nabla_w;

    std::array<int, 2> _output_shape;
public:
    const size_t _size = 0;
    const std::array<int, 3> _shape {0, 0, 0};
    ConvolLayer(std::array<int, 3>);
    void init(size_t n_samples);
    void initParams();

    void fwd();
    void bwd();

    Tensor<float, 4> act(const Tensor<float, 2>&);
    Tensor<float, 4> grad_act(const Tensor<float, 2>&);

    Tensor<float, 4> get_act();
    Tensor<float, 4> get_grad();

    virtual ~ConvolLayer() = default;

    void update(float, float, float);
};
#endif

#endif