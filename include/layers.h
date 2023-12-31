#ifndef LAYERS_H
#define LAYERS_H

#include<Eigen/Dense>
#include<random>

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::Dynamic;

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
    virtual void fwd(const MatrixXf&) = 0;
    virtual void bwd() = 0;
    virtual void bwd(const MatrixXf&) = 0;

    virtual void init(size_t){}
    virtual void initParams(size_t) = 0;

    virtual MatrixXf get_act() = 0;
    virtual MatrixXf get_grad() = 0;

    Layer* next();
    Layer* prev();

    virtual ~Layer() = default;

    virtual void update(float rate, float mu, float size){}; 
};

class InputLayer: public Layer
{
    MatrixXf _act;
public:
    InputLayer(size_t);
    void init(size_t n_samples);
    void initParams(size_t size){}

    void fwd(){}
    void fwd(const MatrixXf&);
    void bwd(){};
    void bwd(const MatrixXf&){};

    virtual MatrixXf get_act();
    virtual MatrixXf get_grad();
};

class FCLayer: public Layer
{
protected:
    MatrixXf _weights;
    VectorXf _biases;

    MatrixXf _winputs; 

    MatrixXf _act;
    MatrixXf _grad;

    MatrixXf _nabla_b;
    MatrixXf _nabla_w;

public:
    FCLayer(size_t size);
    void init(size_t n_samples);
    void initParams(size_t);

    void fwd();
    void bwd();

    void fwd(const MatrixXf&){}
    void bwd(const MatrixXf&);

    virtual MatrixXf act(const MatrixXf&) = 0;
    virtual MatrixXf grad_act(const MatrixXf&) = 0;

    virtual MatrixXf get_act();
    virtual MatrixXf get_grad();

    virtual ~FCLayer() = default;

    void update(float, float, float);
};

class SigmoidLayer: public FCLayer
{
public:
    SigmoidLayer(size_t size); 
    MatrixXf act(const MatrixXf&);
    MatrixXf grad_act(const MatrixXf&);
};

class TanhLayer: public FCLayer
{
public:
    MatrixXf act(const MatrixXf&);
    MatrixXf grad_act(const MatrixXf&);
};

class SoftMaxLayer: public FCLayer
{
public:
    MatrixXf act(const MatrixXf&);
    MatrixXf grad_act(const MatrixXf&);
};

#endif