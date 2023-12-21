#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "activations.h"

using Eigen::Matrix;
using Eigen::Vector;
using Eigen::Dynamic;

template<typename Scalar>
class Sequential
{
public:
    int num_layers{};

    std::vector<Matrix<Scalar, Dynamic, Dynamic>*> weights;
    std::vector<Vector<Scalar, Dynamic>*> activations;

    std::vector<Vector<Scalar, Dynamic>*> w_inputs;
    std::vector<Vector<Scalar, Dynamic>*> biases;

    std::vector<Vector<Scalar, Dynamic>*> nabla_b;
    std::vector<Matrix<Scalar, Dynamic, Dynamic>*> nabla_w;

    std::vector<Vector<Scalar, Dynamic>*> delta;
    ActivationFun<Scalar>* activationFun;

public:
    Sequential() = delete;
    Sequential(std::vector<int> arch, ActivationFun<Scalar>* a)
    {
        activationFun = a;
        num_layers = arch.size();
        int n_last{arch.front()};
        activations.push_back(new Vector<Scalar, Dynamic>(n_last));

        for(std::vector<int>::iterator it {arch.begin() + 1}; it != arch.end(); it++){
            activations.push_back(new Vector<Scalar, Dynamic>(*it));
            w_inputs.push_back(new Vector<Scalar, Dynamic>(*it));

            weights.push_back(new Matrix<Scalar, Dynamic, Dynamic>(*it, n_last));
            weights.back()->setConstant(0.5);

            biases.push_back(new Vector<Scalar, Dynamic>(*it));
            biases.back()->setConstant(0.5);

            nabla_b.push_back(new Vector<Scalar, Dynamic>(*it));
            nabla_w.push_back(new Matrix<Scalar, Dynamic, Dynamic>(*it, n_last));

            delta.push_back(new Vector<Scalar, Dynamic>(*it));

            n_last = *it;
        }
    }

    void feedFwd(const Vector<Scalar, Dynamic>& input){
        size_t n {biases.size()};
        
        // set first layer
        *activations[0] = input;

        *w_inputs[0] = *weights[0] * input + *biases[0];
        *activations[1] = activationFun->activation(*w_inputs[0]);

        for(size_t i{1}; i < n; i++){
            *w_inputs[i] = *weights[i] * *activations[i] + *biases[i];      
            *activations[i+1] = activationFun->activation(*w_inputs[i]);
        }
    }

    void backProp(const Vector<Scalar, Dynamic>& x, 
                 const Vector<Scalar, Dynamic>& y){
        feedFwd(x);

        *delta.back() = cost_grad(y).
            cwiseProduct(activationFun->activation_prime( *(w_inputs.back()) ) );
        *nabla_b.back() = *delta.back();
        *nabla_w.back() = *delta.back() * (*activations[num_layers-2]).transpose();
#if 1
        std::cout << "Sizes: " << delta.size() << " " << num_layers << "\n";
        std::cout << "LOOP" << '\n';
        std::cout << *nabla_b[num_layers-2] << "\n\n"; 
        std::cout << *nabla_w[num_layers-2] << "\n\n";
        for(int i{3}; i < num_layers+1; i++){
            *delta[num_layers-i] = ((*weights[num_layers -i + 1]).transpose() * 
                (*delta[num_layers - i + 1]))
                .cwiseProduct( (activationFun->activation_prime(
                    *w_inputs[num_layers - i])) );

            *nabla_b[num_layers-i] = *(delta[num_layers - i]);
            *nabla_w[num_layers-i] = *(delta[num_layers - i]) * 
                ((*activations[num_layers - i]).transpose());
        }
#endif
    }

    void updateBatch(const std::vector<Vector<Scalar, Dynamic>*>& x, 
                 const std::vector<Vector<Scalar, Dynamic>*>& y, Scalar lr){
        size_t batch_size = x.size();
        for(size_t i{0}; i < batch_size; i++){
            backProp(*x[i], *y[i]);        
        }

        for(int i{0}; i < num_layers-1; i++){
            *weights[i] -= (lr / batch_size) * *nabla_w[i];
            *biases[i] -= (lr / batch_size) * *nabla_b[i];
        }
    }

    Vector<Scalar, Dynamic> cost_grad(const Vector<Scalar, Dynamic>& y){
        return *activations.back() - y;
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