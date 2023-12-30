#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>

#include <Eigen/Dense>

#include "activations.h"
#include "costs.h"

using Eigen::Matrix;
using Eigen::seqN;
using Eigen::all;
using Eigen::last;
using Eigen::Vector;
using Eigen::Dynamic;

static std::random_device rd{};
static std::mt19937 gen{rd()};

class NormalSample
{
    std::normal_distribution<float> nd;
public:
    NormalSample() = delete;
    NormalSample(float, float);
    float operator()();
};

//float normal_distribution(std::normal_distribution<double> nd){
//    return static_cast<float>(nd(gen));
//}

class Sequential
{
public:
    size_t num_layers{};
    std::vector<int> arch;

    std::vector<Matrix<float, Dynamic, Dynamic>> weights;
    std::vector<Vector<float, Dynamic>> biases;

    std::vector<Matrix<float, Dynamic, Dynamic>> activations;
    std::vector<Matrix<float, Dynamic, Dynamic>> w_inputs;

    std::vector<Vector<float, Dynamic>> nabla_b;
    std::vector<Matrix<float, Dynamic, Dynamic>> nabla_w;

    std::vector<Matrix<float, Dynamic, Dynamic>> delta;
    // For momentum
    Matrix<float, Dynamic, Dynamic> cached_weights;
    Vector<float, Dynamic> cached_biases;

    std::vector<ActivationFun<float>*> activationFun;

    CostFun* costFun;

public:
    Sequential() = delete;
    Sequential(std::vector<int>&, ActivationFun<float>*, CostFun*, bool);
    

    void feedFwd(const Matrix<float, Dynamic, Dynamic>& input);

    void backProp(const Matrix<float, Dynamic, Dynamic>& x, 
                 const Matrix<float, Dynamic, Dynamic>& y);

    void initGD(size_t n_samples);

    void GD(Matrix<float, Dynamic, Dynamic>& x,
            Matrix<float, Dynamic, Dynamic>& y, 
            int epochs, float lr, float eta);

    void GD(Matrix<float, Dynamic, Dynamic>& x,
        Matrix<float, Dynamic, Dynamic>& y, 
        int epochs, float lr, float eta,
        Matrix<float, Dynamic, Dynamic>& val_x,
        Matrix<float, Dynamic, Dynamic>& val_y);

    void SGD(Matrix<float, Dynamic, Dynamic>& x,
             Matrix<float, Dynamic, Dynamic>& y, 
            int epochs, int batch_size, float lr, float eta);

    void SGD(Matrix<float, Dynamic, Dynamic>& x,
             Matrix<float, Dynamic, Dynamic>& y, 
            int epochs, int batch_size, float lr, float eta,
            Matrix<float, Dynamic, Dynamic>& val_x,
            Matrix<float, Dynamic, Dynamic>&val_y);

    float accuracy(const Matrix<float, Dynamic, Dynamic>& x,
                    const Matrix<float, Dynamic, Dynamic>& y);

    friend std::ostream& operator<<(std::ostream& out, const Sequential& model);
};


#endif