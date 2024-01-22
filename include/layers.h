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
    const size_t _size = 0;
    Layer(size_t);
    Layer* _next = nullptr;
    Layer* _prev = nullptr;

    virtual void fwd() = 0;
    virtual void fwd(const Tensor<float, 2>&) = 0;
    virtual void bwd() = 0;
    virtual void bwd(const Tensor<float, 2>&) = 0;

    virtual void init(size_t){}
    virtual void initParams(size_t) = 0;

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
    InputLayer(size_t);
    void init(size_t n_samples);
    void initParams(size_t size){}

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
    FCLayer(size_t size);
    void init(size_t n_samples);
    void initParams(size_t);

    void fwd();
    void bwd();

    void fwd(const Tensor<float, 2>&){}
    void bwd(const Tensor<float, 2>&);

    virtual Tensor<float, 2> act(const Tensor<float, 2>&) = 0;
    virtual Tensor<float, 2> grad_act(const Tensor<float, 2>&) = 0;

    virtual Tensor<float, 2> get_act();
    virtual Tensor<float, 2> get_grad();

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

#endif