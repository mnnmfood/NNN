#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

#include <Eigen/Dense>

#include "activations.h"

using Eigen::Matrix;
using Eigen::seqN;
using Eigen::all;
using Eigen::last;
using Eigen::Vector;
using Eigen::Dynamic;

template<typename Scalar>
class Sequential
{
public:
    int num_layers{};
    std::vector<int> arch;

    std::vector<Matrix<Scalar, Dynamic, Dynamic>*> weights;
    std::vector<Vector<Scalar, Dynamic>*> biases;

    std::vector<Matrix<Scalar, Dynamic, Dynamic>*> activations;
    std::vector<Matrix<Scalar, Dynamic, Dynamic>*> w_inputs;

    std::vector<Vector<Scalar, Dynamic>*> nabla_b;
    std::vector<Matrix<Scalar, Dynamic, Dynamic>*> nabla_w;

    std::vector<Matrix<Scalar, Dynamic, Dynamic>*> delta;

    ActivationFun<Scalar>* activationFun;

public:
    Sequential() = delete;
    Sequential(std::vector<int>& arch_in, ActivationFun<Scalar>* a)
    {
        arch = arch_in;
        activationFun = a;
        num_layers = arch.size();
        int n_last{arch.front()};

        for(std::vector<int>::iterator it {arch.begin() + 1}; it != arch.end(); it++){

            weights.push_back(new Matrix<Scalar, Dynamic, Dynamic>(*it, n_last));
            weights.back()->setRandom();

            biases.push_back(new Vector<Scalar, Dynamic>(*it));
            biases.back()->setRandom();

            nabla_b.push_back(new Vector<Scalar, Dynamic>(*it));
            nabla_w.push_back(new Matrix<Scalar, Dynamic, Dynamic>(*it, n_last));

            n_last = *it;
        }
    }

    void feedFwd(const Matrix<Scalar, Dynamic, Dynamic>& input){
        size_t n {biases.size()};
        
        // set first layer
        *activations[0] = input;

        *w_inputs[0] = *weights[0] * input + *biases[0];
        *activations[1] = activationFun->activation(*w_inputs[0]);

        for(size_t i{1}; i < num_layers-1; i++){
            *w_inputs[i] = *weights[i] * *activations[i] + *biases[i];      
            *activations[i+1] = activationFun->activation(*w_inputs[i]);
        }
    }

    void backProp(const Matrix<Scalar, Dynamic, Dynamic>& x, 
                 const Matrix<Scalar, Dynamic, Dynamic>& y){
        feedFwd(x);

        *delta.back() = cost_grad(y).
            cwiseProduct(activationFun->activation_prime( *(w_inputs.back()) ) );
        *nabla_b.back() = *delta.back();
        *nabla_w.back() = *delta.back() * (*activations[num_layers-2]).transpose();
#if 0
        for(int i{3}; i < num_layers+1; i++){
            *delta[num_layers-i] = ((*weights[num_layers -i + 1]).transpose() * 
                (*delta[num_layers - i + 1]))
                .cwiseProduct( (activationFun->activation_prime(
                    *w_inputs[num_layers - i])) );

            //*nabla_b[num_layers-i] += *(delta[num_layers - i]);
            //*nabla_w[num_layers-i] += *(delta[num_layers - i]) * 
            //    ((*activations[num_layers - i]).transpose());
        }
#endif
    }

    void initSGD(size_t n_samples){
        activations.push_back(
            new Matrix<Scalar, Dynamic, Dynamic>(arch[0], n_samples));
        for(int i{1}; i < num_layers; i++){
            delta.push_back(
                new Matrix<Scalar, Dynamic, Dynamic>(arch[i], n_samples));
            activations.push_back(
                new Matrix<Scalar, Dynamic, Dynamic>(arch[i], n_samples));
            w_inputs.push_back(
                new Matrix<Scalar, Dynamic, Dynamic>(arch[i], n_samples));
        }
    }

    void SGD(Matrix<Scalar, Dynamic, Dynamic>& x,
             Matrix<Scalar, Dynamic, Dynamic>& y, 
            int epochs, int batch_size, Scalar lr){

        size_t train_size = x.cols();
        initSGD(train_size);

        // Prepare random indices generator
        std::random_device rd;
        std::mt19937 g(rd());
        std::vector<int> indices;
        for(int i{0}; i < train_size; i++){indices.push_back(i);} 

        double cost_t;

        for(int k{0}; k < epochs; k++){
            cost_t = 0;

            // Get random subsample
            std::shuffle(indices.begin(), indices.end(), g);
            for(size_t i{0}; i < batch_size; i++){
                backProp(x(all, indices[i]), y(all, indices[i]));        
            }

            for(int i{0}; i < num_layers-1; i++){
                *weights[i] -= (lr / batch_size) * *nabla_w[i];
                *biases[i] -= (lr / batch_size) * *nabla_b[i];
            }
            
            for(size_t i{1}; i < batch_size; i++){
                feedFwd(*x[indices[train_size - i]]);
                cost_t += cost(*y[indices[train_size - i]]);
            }
            std::cout << "Cost " << k << " :" << cost_t / batch_size << "\n";
        }
    }

    Vector<Scalar, Dynamic> cost_grad(const Vector<Scalar, Dynamic>& y){
        return *activations.back() - y;
    } 

    Scalar cost(const Vector<Scalar, Dynamic>& y){
        return (*activations.back() - y).squaredNorm();
    }

    Scalar accuracy(const std::vector<Vector<Scalar, Dynamic>*>& x, 
                    const std::vector<Vector<Scalar, Dynamic>*>& y){
        size_t test_size{x.size()};
        std::cout << "Test size " << test_size << "\n";
        int pred, test; 
        int sum{0};
        for(size_t i{0}; i < test_size; i++){
            feedFwd(*x[i]);
            activations.back()->maxCoeff(&pred);
            y[i]->maxCoeff(&test);
            std::cout << "Test: " << i << " :" << test << " Prediction: " << pred << "\n";

            sum += (pred==test);
        }
        return static_cast<Scalar>(sum) / static_cast<Scalar>(test_size);
    }

    Vector<Scalar, Dynamic> getLayer(int i){
        return *activations[i];
    }

    friend std::ostream& operator<<(std::ostream& out, const Sequential& model){
        out << "Layer" << '\t' << "Nodes" << '\t' << "Weights" << '\n';
        size_t l = model.weights.size();
        out << 0 << '\t' << model.weights[0]->cols() << '\t' << 0 << '\n';
        for(int i{0}; i < l; i++){
            out << i+1 << '\t' << model.biases[i]->size() << 
            '\t' << model.weights[i]->size() << '\n';
        }
        return out;
    } 
};


#endif