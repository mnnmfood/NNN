#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include <Eigen/Dense>
#include <vector>
#include <initializer_list>
#include "layers.h"
#include "costs.h"
#include "typedefs.h"

class Sequential2
{

    std::vector<Layer*> _layers;
    CostFun* _cost;

public:
    const size_t num_layers;
    Sequential2(std::initializer_list<Layer*> layers, CostFun* cost);
    void init(size_t n_samples);
    void bkwProp(const Tensor<float, 2>&);
    void fwdProp(const Tensor<float, 2>&);
    void SGD(Tensor<float, 2>& x,
            Tensor<float, 2>& y, 
            int epochs, int batch_size, float lr, float mu,
            Tensor<float, 2>& val_x,
            Tensor<float, 2>&val_y);
    float accuracy(const Tensor<float, 2>&x, const Tensor<float, 2>& y);
    ~Sequential2();


};

#endif