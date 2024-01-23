#ifndef LAYERS_H
#define LAYERS_H

#include<Eigen/Dense>
#include<random>
#include "typedefs.h"

inline std::random_device rd{};
inline std::mt19937 gen{rd()};

class Layer
{
public:
    Layer(size_t);
    Layer* _next = nullptr;
    Layer* _prev = nullptr;

    virtual void fwd() = 0;
    virtual void fwd(const Tensor<float, 2>&) = 0;
    virtual void bwd() = 0;
    virtual void bwd(const Tensor<float, 2>&) = 0;

    virtual void init(size_t){}
    virtual void initParams() = 0;

    virtual Tensor<float, 2> get_act() = 0;
    virtual Tensor<float, 2> get_grad() = 0;

    Layer* next();
    Layer* prev();

    virtual ~Layer() = default;

    virtual void update(float rate, float mu, float size){}; 
};

class InputLayer: public Layer
{
    Tensor<float, 2> _act;
public:
    const size_t _size = 0;
    InputLayer(size_t);
    void init(size_t n_samples);
    void initParams(){}

    void fwd(){}
    void fwd(const Tensor<float, 2>&);
    void bwd(){};
    void bwd(const Tensor<float, 2>&){};

    virtual Tensor<float, 2> get_act();
    virtual Tensor<float, 2> get_grad();
};

class FCLayer: public Layer
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
    const size_t _size = 0;
    FCLayer(size_t size);
    void init(size_t n_samples);
    void initParams();

    void fwd();
    void bwd();

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