#ifndef LAYERS_H
#define LAYERS_H

#include<Eigen/Dense>
#include<random>
#include "Tensor.h"
#include "typedefs.h"
#include "eigenFuns.h"
#include "layer_traits.h"

inline const std::array<int, 1> dims_colwise {0};
inline const std::array<int, 1> dims_rowwise {1};

inline std::random_device rd{};
inline std::mt19937 gen{rd()};

class BaseLayer
{
public:
    const size_t _out_num_dims;
    const size_t _in_num_dims;
    BaseLayer* _next = nullptr;
    BaseLayer* _prev = nullptr;

    BaseLayer* next();
    BaseLayer* prev();
    BaseLayer(size_t, size_t);

    virtual void fwd() = 0;
    virtual void fwd(TensorWrapper<float>&&) = 0;
    virtual void bwd() = 0;
    virtual void bwd(TensorWrapper<float>&&) = 0;

    virtual void init(Index) = 0;
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
    using out_batch_shape_t = std::array<Index, 
                std::tuple_size<out_shape_t>{} + 1>;
    using in_batch_shape_t = std::array<Index, 
                std::tuple_size<in_shape_t>{} + 1>;
    const size_t out_dims {std::tuple_size<out_shape_t>{}};
    const size_t in_dims {std::tuple_size<in_shape_t>{}};
    const size_t num_dims {traits<Derived>::NumDimensions};
    using out_t = Tensor<float, std::tuple_size<out_shape_t>{}+1>;
    using in_t = Tensor<float, std::tuple_size<in_shape_t>{}+1>;
    using weight_t = Tensor<float, traits<Derived>::NumDimensions>;
    using bias_t = Tensor<float, 1>;
    using nabla_weight_t = Tensor<float, traits<Derived>::NumDimensions>;
    using nabla_b_t = Tensor<float, 2>;

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
    out_batch_shape_t _out_batch_shape;
    in_batch_shape_t _in_batch_shape;

public:
    Layer(out_shape_t out_shape)
        :BaseLayer {out_dims, in_dims}, _out_shape{out_shape}
    {
        std::copy(_out_shape.begin(), _out_shape.end(), 
            _out_batch_shape.begin());
    }    
    Layer(out_shape_t out_shape, in_shape_t in_shape)
        :BaseLayer {out_dims, in_dims}, _out_shape{out_shape},
        _in_shape{in_shape}
    {
        std::copy(_out_shape.begin(), _out_shape.end(), 
            _out_batch_shape.begin());

        std::copy(_in_shape.begin(), _in_shape.end(), 
            _in_batch_shape.begin());
    }    
    TensorWrapper<float> get_act(){
        return TensorWrapper(_act);
    }
    TensorWrapper<float> get_grad(){
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
    in_shape_t prev_shape(){
        return _prev->out_shape().get<in_shape_t>();
    }
    out_shape_t next_shape(){
        return _next->in_shape().get<out_shape_t>();
    }
};

template<size_t N>
class InputLayer: public Layer<InputLayer<N>>
{
    typedef typename traits<InputLayer<N>>::out_shape_t out_shape_t;
    using out_t = Tensor<float, std::tuple_size<out_shape_t>{}+1>;
    std::array<Index, N> _shape;
    std::array<Index, N+1> batch_shape;
public:
    const size_t _size = 0;
    InputLayer(std::array<Index, N> shape):
        Layer<InputLayer<N>>{shape, shape} ,_shape{shape}
    {
    }
    void init(Index n_samples){
        this->_out_batch_shape.back() = n_samples;
        this->_in_batch_shape.back() = n_samples;
        this->_act = out_t(this->_out_batch_shape);
    }
    void initParams(){}
    void fwd(){}
    void fwd(TensorWrapper<float>&& input){
        this->_act = input.get(this->_in_batch_shape);
    }
    void bwd(){};
    void bwd(TensorWrapper<float>&& output){};
};

class FCLayer: public Layer<FCLayer>
{
    std::array<Index, 1> _shape;
public:
    FCLayer(Index size);
    void init(Index batch_size);
    void initParams();

    void fwd();
    void bwd();

    void fwd(TensorWrapper<float>&&);
    void bwd(TensorWrapper<float>&&);

    virtual Tensor<float, 2> act(const Tensor<float, 2>&) = 0;
    virtual Tensor<float, 2> grad_act(const Tensor<float, 2>&) = 0;

    virtual ~FCLayer() = default;
};

class SigmoidLayer: public FCLayer
{
public:
    SigmoidLayer(Index size); 
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

#endif
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
