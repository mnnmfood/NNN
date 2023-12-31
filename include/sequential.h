#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include <Eigen/Dense>
#include <vector>
#include <initializer_list>
#include "layers.h"
#include "costs.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::all;

class Sequential2
{

    std::vector<Layer*> _layers;
    CostFun* _cost;

public:
    const size_t num_layers;
    Sequential2(std::initializer_list<Layer*> layers, CostFun* cost);
    void init(size_t n_samples);
    void bkwProp(const MatrixXf&);
    void fwdProp(const MatrixXf&);
    void SGD(Matrix<float, Dynamic, Dynamic>& x,
            Matrix<float, Dynamic, Dynamic>& y, 
            int epochs, int batch_size, float lr, float mu,
            Matrix<float, Dynamic, Dynamic>& val_x,
            Matrix<float, Dynamic, Dynamic>&val_y);
    float accuracy(const MatrixXf&x, const MatrixXf& y);
    ~Sequential2();


};

#endif