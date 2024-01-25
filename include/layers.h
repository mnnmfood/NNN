#ifndef LAYERS_H
#define LAYERS_H

#include <array>
#include<Eigen/Dense>
#include<random>
#include "typedefs.h"

inline std::random_device rd{};
inline std::mt19937 gen{rd()};

class BaseLayer
{
public:
    BaseLayer(int);
    BaseLayer* _next = nullptr;
    BaseLayer* _prev = nullptr;

    virtual void fwd() = 0;
    virtual void bwd() = 0;

    virtual void init(size_t){}
    virtual void initParams() = 0;

    BaseLayer* next();
    BaseLayer* prev();

    virtual ~BaseLayer() = default;

    virtual void update(float rate, float mu, float size){}; 
    int _depth = 0;
};

template<int N>
class Layer: public BaseLayer
{
public:
    std::array<Index, N> _shape;
    std::array<Index, N> _output_shape;
    Layer(std::array<Index, N>& shape)
        :BaseLayer{N}, _shape{shape}
    {
    }
    Layer(std::array<Index, N>&& shape)
        :BaseLayer{N}, _shape{shape}
    {
    }
    virtual void fwd() = 0;
    virtual void fwd(const Tensor<float, N+1>&) = 0;
    virtual void bwd() = 0;
    virtual void bwd(const Tensor<float, N+1>&) = 0;
    virtual void init(size_t){}
    virtual void initParams() = 0;
    virtual Tensor<float, N+1> get_act() = 0;
    virtual Tensor<float, N+1> get_grad() = 0;
    virtual void update(float rate, float mu, float size){}; 
    virtual ~Layer() = default;
};

template<size_t N>
class InputLayer: public Layer<N>
{
    Tensor<float, N+1> _act;
public:
    InputLayer(std::array<Index, N> shape)
        :Layer<N>(shape)
    {
    }
    void init(Index n_samples){
        std::array<Index, N+1> temp;
        std::copy_n(this->_shape.begin(), N, temp);
        temp.back() = n_samples;
        _act = Tensor<float, N+1> (temp);
    }
    void initParams(){}

    void fwd(){}
    void fwd(const Tensor<float, N+1>& input){
        _act = input;
    }
    void bwd(){};
    void bwd(const Tensor<float, N+1>&){};

    Tensor<float, N+1> get_act(){
        return _act;
    }
    Tensor<float, N+1> get_grad(){
        return _act;
    }
};

class FCLayer: public Layer<1>
{
protected:
    Tensor<float, 2> _weights;
    Tensor<float, 1> _biases;

    Tensor<float, 2> _winputs; 

    Tensor<float, 2> _act;
    Tensor<float, 2> _grad;

    Tensor<float, 2> _nabla_b;
    Tensor<float, 2> _nabla_w;

public:
    FCLayer(Index size);
    void init(Index n_samples);
    void initParams();

    void fwd();
    void bwd();
    void fwd(const Tensor<float, 2>&);
    void bwd(const Tensor<float, 2>&);

    virtual Tensor<float, 2> act(const Tensor<float, 2>&) = 0;
    virtual Tensor<float, 2> grad_act(const Tensor<float, 2>&) = 0;

    Tensor<float, 2> get_act();
    Tensor<float, 2> get_grad();

    virtual ~FCLayer() = default;

    void update(float, float, float);
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
    TanhLayer(Index size);
    Tensor<float, 2> act(const Tensor<float, 2>&);
    Tensor<float, 2> grad_act(const Tensor<float, 2>&);
};

class SoftMaxLayer: public FCLayer
{
public:
    SoftMaxLayer(Index size);
    Tensor<float, 2> act(const Tensor<float, 2>&);
    Tensor<float, 2> grad_act(const Tensor<float, 2>&);
};

class ConvolLayer:public Layer<3>
{
protected:
    Tensor<float, 3> _weights;
    Tensor<float, 1> _biases;

    Tensor<float, 2> _winputs; 

    Tensor<float, 4> _act;
    Tensor<float, 4> _grad;

    Tensor<float, 2> _nabla_b;
    Tensor<float, 3> _nabla_w;

public:
    ConvolLayer(std::array<Index, 3>);
    void init(Index n_samples);
    void initParams();

    void fwd();
    void bwd();
    void fwd(const Tensor<float, 4>&);
    void bwd(const Tensor<float, 4>&);

    Tensor<float, 4> act(const Tensor<float, 4>&);
    Tensor<float, 4> grad_act(const Tensor<float, 4>&);

    Tensor<float, 4> get_act();
    Tensor<float, 4> get_grad();

    virtual ~ConvolLayer() = default;

    void update(float, float, float);
};

#endif