#ifndef COSTS_H
#define COSTS_H

#include <cmath>
#include <iostream>
#include <functional>
#include "typedefs.h"
#include "layer_traits.h"
#include "Tensor.h"
#include "cost_funs.h"


// -- Base clases
class CostFun
{
public:
    virtual Tensor<float, 0> cost(TensorWrapper<float>& a,
                        TensorWrapper<float>& y, ThreadPoolDevice*) = 0;
    virtual void grad(
        TensorWrapper<float>& a, TensorWrapper<float>& y, 
        TensorWrapper<float>& grad, ThreadPoolDevice*) = 0;

    virtual void act(TensorWrapper<float>& z, 
        TensorWrapper<float>& act, ThreadPoolDevice*) = 0;

    virtual void init(TensorShape&& shape) = 0;
    virtual ~CostFun() = default;
};

template<class Derived>
class CostFunTempl : public CostFun
{
protected:
    typedef typename traits<Derived>::shape_t shape_t;
    using tensor_t = Tensor<float, 
                std::tuple_size<shape_t>{}>;
    using tensormap_t = TensorMap<Tensor<float, 
                std::tuple_size<shape_t>{}>>;
    shape_t _shape;
public:
    tensormap_t get(TensorWrapper<float>& t) { 
        return t.get(_shape); 
    }

    Tensor<float, 0> cost(TensorWrapper<float>& a, 
        TensorWrapper<float>& y, ThreadPoolDevice* device) override{
        return static_cast<Derived*>(this)->cost(get(a), get(y), device);
    }

    void grad(TensorWrapper<float>& a, TensorWrapper<float>& y, 
        TensorWrapper<float>& grad, ThreadPoolDevice* device) override{
        static_cast<Derived*>(this)->grad(get(a), get(y), get(grad), device);
    }

    void act(TensorWrapper<float>& z, TensorWrapper<float>& act,
        ThreadPoolDevice* device) override{
        static_cast<Derived*>(this)->act(get(z), get(act), device);
    }

    void init(TensorShape&& shape) {
        _shape = shape_t(shape.get<shape_t>());
    }
};

template<size_t N>
class DummyCost : public CostFunTempl<DummyCost<N>>
{
    typedef TensorMap<Tensor<float, N + 1>> tensormap_t;
 
public:
    Tensor<float, 0> cost(tensormap_t a,
        tensormap_t y, ThreadPoolDevice*){
        throw std::runtime_error("A cost function is needed for training");
    }

    void grad(tensormap_t a, tensormap_t y, 
        tensormap_t grad, ThreadPoolDevice*) {
        throw std::runtime_error("A cost function is needed for training");
    }

    void act(tensormap_t z, tensormap_t act, 
        ThreadPoolDevice*) {
        act = z;
    }
};

// -- Implementations

class MSE : public CostFunTempl<MSE>
{
    typedef TensorMap<Tensor<float, 2>> tmap_t;
public:
    Tensor<float, 0> cost(tmap_t a,
        tmap_t y, ThreadPoolDevice*);

    void grad(tmap_t a, tmap_t y, 
        tmap_t grad, ThreadPoolDevice*);

    void act(tmap_t z, tmap_t act, 
        ThreadPoolDevice*);
};

class CrossEntropy : public CostFunTempl<CrossEntropy>
{
    typedef TensorMap<Tensor<float, 2>> tmap_t;
    bool _softmax;
public:
    CrossEntropy(bool softmax=true) :_softmax{ softmax } {}
    Tensor<float, 0> cost(tmap_t a,
        tmap_t y, ThreadPoolDevice*);

    void grad(tmap_t a, tmap_t y, 
        tmap_t grad, ThreadPoolDevice*);

    void act(tmap_t z, tmap_t act, 
        ThreadPoolDevice*);
};

#endif
